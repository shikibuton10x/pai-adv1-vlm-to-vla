"""REST-only bounded Gemini client. No SDK continuation or automatic retries."""
import base64
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from .budget import INPUT_CAP, OUTPUT_CAP, INPUT_MICRO, OUTPUT_MICRO, BudgetStop

from .inference_profile import MEDIUM,certified

MODEL = 'models/gemini-4-argon'
HTTP_TIMEOUT_S = 90
SYSTEM = '''You operate a simulated robot for a grasp-and-lift connection test.
The task is to grasp the wine bottle and lift it by at least 20mm, then hold it.
The task's original rack instruction is context; do not attempt rack placement yet.
Agent image shows EEF + world X(red),Y(green),Z(blue) axes, each 50mm long.
Images are natural camera views: vertical flip of raw simulator images, once.
EEF translation units: world-frame metres. Rotation: world-frame axis-angle radians,
left-multiplied onto the current EEF rotation. Never output executable code.
Return exactly one JSON object. Allowed commands:
{"operation":"move_delta","delta_world_m":[0,0,0.02],"steps":20}
{"operation":"rotate_delta","rotation_world_rad":[0,0,0.1],"steps":20}
{"operation":"open","steps":10} or {"operation":"close","steps":10}
{"operation":"hold","steps":10} or {"operation":"stop"}.
All vector components must be finite numbers; steps integer1..20.
The executor proportionally rescales an oversized finite translation vector to
norm0.06m or a rotation vector to norm0.35rad, preserving direction and steps.
Feedback shows requested/applied/clipped, actual motion, and whether the target
was unreached. Do not assume the full requested displacement was executed.
Moves retain the latest gripper command. Close is not proof of grasp.
After every short command you receive fresh images and EEF/gripper state.
Avoid contact with other objects, the rack, and supports. Stop if uncertain.
Do not return prose, markdown, unknown fields, multiple commands or object coordinates.'''


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def request_hash(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def build_request(payload, image_paths, *, system_suffix='', reference_image_path=None,current_side_image_path=None):
    if len(image_paths) != 2:
        raise ValueError('Exactly two current views are required')
    if reference_image_path is not None and current_side_image_path is not None:
        raise ValueError('Past reference and current-side variants are mutually exclusive')
    text = json.dumps(payload, sort_keys=True, allow_nan=False)
    if len(text.encode()) > 12000:
        raise ValueError('Robot/history payload too large')
    parts = [{'text': text}]
    for path in image_paths:
        raw = Path(path).read_bytes()
        if len(raw) > 500_000:
            raise ValueError('Image input byte cap exceeded')
        parts.append({'inlineData':{'mimeType':'image/png','data':base64.b64encode(raw).decode()}})
    if reference_image_path is not None:
        raw=Path(reference_image_path).read_bytes()
        if len(raw)>500_000:raise ValueError('Reference image input byte cap exceeded')
        parts.extend([{'text':'PAST REFERENCE IMAGE ONLY: agentview at control step0 before the source approach. This is NOT the current state. Current views are the preceding two images.'},
            {'inlineData':{'mimeType':'image/png','data':base64.b64encode(raw).decode()}}])
    if current_side_image_path is not None:
        raw=Path(current_side_image_path).read_bytes()
        if len(raw)>500_000:raise ValueError('Current side image input byte cap exceeded')
        parts.extend([{'text':'CURRENT fixed sideview RGB: same frozen simulator state as the preceding agentview and wrist images. This is not a past reference.'},
            {'inlineData':{'mimeType':'image/png','data':base64.b64encode(raw).decode()}}])
    return dict(model=MODEL, systemInstruction={'parts':[{'text':SYSTEM+system_suffix}]},
        contents=[{'role':'user','parts':parts}],
        generationConfig={'responseMimeType':'application/json',
            'maxOutputTokens':OUTPUT_CAP, 'thinkingConfig':{'thinkingLevel':'MEDIUM'}})


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *unused, **kwargs):
        raise RuntimeError('Redirect refused')


class RestTransport:
    """Credentials are obtained only at an explicitly selected live send."""
    def __init__(self,*,profile=MEDIUM):
        self.profile=certified(profile)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def post(self, method, request):
        if method == 'countTokens':
            body = {'generateContentRequest':request}
        elif method == 'generateContent':
            body = {k:v for k,v in request.items() if k != 'model'}
        else:
            raise ValueError('Unsupported endpoint')
        key = os.environ.get('GEMINI_API_KEY')
        # Match the prior client's plain, non-whitespace credential-file format.
        # Reject header control characters without assuming a particular key alphabet.
        if not key or len(key)>512 or any(not 33<=ord(char)<=126 for char in key):
            raise RuntimeError('GEMINI_API_KEY absent or malformed; value never logged')
        req = urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/'+MODEL+':'+method,
            data=encoded(body), headers={'Content-Type':'application/json','x-goog-api-key':key},method='POST')
        try:
            with self.opener.open(req, timeout=self.profile.http_timeout_s) as response:
                raw = response.read(2_000_001)
            if len(raw)>2_000_000:
                raise RuntimeError('Provider response byte cap exceeded')
            return json.loads(raw)
        except urllib.error.HTTPError as error:
            # No header, URL, or provider error-body logging (may contain secrets).
            raise RuntimeError('Provider HTTP status '+str(error.code)) from None
        except urllib.error.URLError:
            raise RuntimeError('Provider network failure; reservation retained') from None


def token_integer(value):
    if type(value) is not int or value < 0:
        raise ValueError('Invalid/missing usage token count')
    return value


class Client:
    def __init__(self, transport, ledger, out,*,profile=MEDIUM):
        self.profile=certified(profile)
        if isinstance(transport,RestTransport) and transport.profile!=self.profile:raise ValueError('HTTP/inference profile mismatch')
        self.transport, self.ledger, self.out = transport, ledger, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)

    def choose(self, request):
        expected = self.profile.generation_config()
        if (set(request) != {'model','systemInstruction','contents','generationConfig'} or
                request['model'] != MODEL or request['generationConfig'] != expected):
            raise BudgetStop('Request model/config differs from reserved hard-cap contract')
        canonical = encoded(request)  # freeze exactly the system/images/history counted
        frozen = json.loads(canonical)
        digest = request_hash(frozen)
        # Official Billing FAQ + Google forum clarification: REST countTokens is
        # free, including generateContentRequest. It still consumes a local call slot.
        count_id = self.ledger.reserve('count', 0, digest)
        (self.out/(count_id+'_request.json')).write_bytes(canonical)
        count = self.transport.post('countTokens', frozen)
        n = token_integer(count.get('totalTokens'))
        self.ledger.settle(count_id, 0, evidence={'totalTokens':n,'billing':'countTokens free'})
        if not 0 < n <= INPUT_CAP:
            raise BudgetStop('Input including system/images exceeds 8000 tokens')
        if request_hash(frozen) != digest:
            raise BudgetStop('Counted request mutated before generation')
        gen_id = self.ledger.reserve('generate', n*INPUT_MICRO+self.profile.output_cap*OUTPUT_MICRO, digest)
        response = self.transport.post('generateContent', frozen)
        (self.out/(gen_id+'_response.json')).write_bytes(encoded(response))
        usage = response.get('usageMetadata', {})
        prompt = token_integer(usage.get('promptTokenCount'))
        candidates = token_integer(usage.get('candidatesTokenCount'))
        thoughts = token_integer(usage.get('thoughtsTokenCount', 0))
        self.ledger.settle(gen_id, prompt*INPUT_MICRO+(candidates+thoughts)*OUTPUT_MICRO,
            evidence={'promptTokenCount':prompt,'candidatesTokenCount':candidates,'thoughtsTokenCount':thoughts})
        if prompt>INPUT_CAP or candidates+thoughts>self.profile.output_cap:
            raise BudgetStop('Provider token contract violated')
        candidate = (response.get('candidates') or [{}])[0]
        finish = candidate.get('finishReason')
        text = ''.join(p.get('text','') for p in candidate.get('content',{}).get('parts',[])
                       if not p.get('thought',False))
        receipt = dict(request_hash=digest, count_reservation=count_id,
            generation_reservation=gen_id, counted_input_tokens=n,
            finish_reason=finish, usage=usage, output_and_thinking_cap=self.profile.output_cap)
        (self.out/(gen_id+'_receipt.json')).write_bytes(encoded(receipt))
        if finish != 'STOP' or not text or len(text.encode())>8192:
            return None, dict(receipt, rejected='NON_STOP_OR_EMPTY_OR_OVERSIZED_OUTPUT',
                rejected_text=text[:2048],rejected_text_truncated=len(text)>2048)
        try:
            def reject_nonfinite(value):
                raise ValueError('Non-finite JSON constant rejected')
            value = json.loads(text,parse_constant=reject_nonfinite)
        except (ValueError, TypeError):
            return None, dict(receipt, rejected='INVALID_JSON',
                rejected_text=text[:2048],rejected_text_truncated=len(text)>2048)
        return value, receipt


class MockTransport:
    def __init__(self, commands):
        self.commands = iter(commands)
        self.counted_hashes = []

    def post(self, method, request):
        if method == 'countTokens':
            self.counted_hashes.append(request_hash(request))
            return {'totalTokens':500}
        if method != 'generateContent' or request_hash(request) != self.counted_hashes[-1]:
            raise AssertionError('Mock generation differs from counted request')
        return {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(next(self.commands))}]}}],
            'usageMetadata':{'promptTokenCount':500,'candidatesTokenCount':80,'thoughtsTokenCount':100}}
