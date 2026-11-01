"""Only pre-authorized inference profiles; physical/prompt contracts live elsewhere."""
from dataclasses import dataclass
@dataclass(frozen=True)
class InferenceProfile:
    thinking_level:str
    output_cap:int
    http_timeout_s:int
    outer_timeout_s:int
    def generation_config(self):return dict(responseMimeType='application/json',maxOutputTokens=self.output_cap,thinkingConfig={'thinkingLevel':self.thinking_level})
MEDIUM=InferenceProfile('MEDIUM',4096,90,1200)
HIGH=InferenceProfile('HIGH',32768,900,1800)
def certified(value):
    if value not in (MEDIUM,HIGH):raise ValueError('Unapproved inference profile')
    return value
