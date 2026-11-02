# 汎用VLMの実演を、小型ロボット方策へ

Physical AI 応用編1 最終課題 / sabia0080 / 2026-11-02

Gemini 4 Argonを使った閉ループの実演収集、SmolVLAへの追加学習、位置・高さだけの診断補助を調べたケーススタディです。主タスクは公開LIBEROの `libero_goal / task 2 / init 0`（瓶を収納棚の平らな天板に置く）です。

## 主な成果

- 採用教師S14D：配置・把持解放・退避後の安定までの実演を収集。人間の方針補足・過去区間の再生を含みます。
- S13：117stepだけXYを補助した後、学生が自ら解放・退避し5秒安定。
- S16：237stepだけZを補助した後、学生が自ら解放・退避し5秒安定。接地前の一時的な把持喪失・再接触があるため、追加の全行程品質判定は未達のままです。
- S0は元の公開モデルで同initを無補助完遂。S22/2000には搬送姿勢の改善が目視で見られましたが、無補助完遂は未達でした。
- S23の結果は `results_summary.json` とレポートで明示します。未実行・失敗を成功へ置き換えません。

補助はシミュレータの内部量を利用し、全行程の無補助成功や汎化性能を示すものではありません。補助中もgripper・回転を強制せず、補助終了後は同じ学生に任せました。中央配置は追加の品質目標です。

## すぐに確認できるもの（GPU/API不要）

```sh
python3 reproduce.py
```

配布済み小記録とソースのSHA、主要な結果の区分を検査します。これはロボットの再走行や訓練の再実行ではありません。動画は提出用に2倍速、API待機・開始前静定を省略しています。20Hzは低レベル制御周波数であり、教師が実時間20Hzで判断するという意味ではありません。

## コードの入口

|機能|実装|
|---|---|
|環境・保全計測|`src/course_award/env.py`, `safety.py`|
|教師操作・記録|`src/course_award/gemini.py`, `teacher_control.py`, `scripts/run_s14_approach.py`（対応する設定を参照）|
|水平位置のみ補助|`src/course_award/xy_only_s13.py` とS13の実行スクリプト|
|高さのみ補助|`src/course_award/height_only_s16.py` とS16の実行スクリプト|
|S22追加学習|`scripts/train_s22.py`, `src/course_award/learn_s22.py`|
|S23追加学習|`scripts/train_s23.py`, `src/course_award/learn_s23.py`|
|学生評価|各 `scripts/run_s*_student.py` と `evaluation_s11.py`|

全科学コードと設定を収録しています。`scripts/`内の運用実験入口は、記録当時の入力manifestや小fixtureを参照するものがあり、この公開パッケージだけでは全て起動できません。クラウド接続・個人専用起動・認証情報・巨大なモデル重み・カメラ全フレーム・他チームのデータは含めません。`EXPORT_MANIFEST.json`が収録/除外/パス置換を示します。

## 学習・ロールアウトの再実行に必要なもの

1. 公開LIBERO資産と、実験で使ったMuJoCo/robosuite/runtime（版は各config）。LeRobotはv0.6.1。Linux GPU/EGL環境を推奨します。
2. 公開 `lerobot/smolvla_libero` revision `a24f1f380461b665158d52acd140152c79e8b36d` と、`configs/s2_acquisition_manifest.json`のbackbone/processor。
3. `local_assets.example.json`を自分の環境のパスで記入。保存済み物理状態・入力照合fixture・教師1028組の画像/state/action、および評価する追加学習checkpointが別途必要です。これらの大容量研究記録は本人のprivate保管にあり、このrepoに公開配布していません。
4. 同梱設定中の `LOCAL_PATH_REQUIRED` や `YOUR_ACCOUNT/PRIVATE_ARCHIVE` は環境別の値に置換する明示的placeholderです。これは学習値の変更ではありません。
5. 教師を新たに動かす場合は、利用するAPIの認証と利用料金が別途必要。保存actionの再生とは区別してください。

完全な再走行に必要な私有入力を同梱していないので、第三者がこのrepoだけで全結果を再現できるとは主張しません。公開元モデルや環境を改変せず、まず入力/版/境界一致を確認する設計です。

## 学習・評価の主要条件

1028組の採用教師データ（observer除外）、chunk50、micro4×acc4、expert/state/action/time projection更新、既存正規化を保持。S22はFP32 trainableとBF16 autocast、peak2e-5→2e-6、2000更新。S23はS22/2000のFP32保存値からfresh AdamW、peak1e-4、horizon2000の前半1000更新へ実行前に縮小。末尾LRはfloor2e-6にまだ達しない。実行の有無は結果表を参照。評価n_action_steps10、同init0、raw7D/state8D/2RGB。S22/23の保存点は独立した学習seedではありません。

## 出典とライセンス

自作・既存本人コードの利用条件はApache-2.0。第三者のLIBEROはMIT、LeRobot/SmolVLAは各公開元のライセンスを参照。第三者コードやモデルを本ライセンスで再許諾するものではありません。`NOTICE.md`とprovenanceに再利用元・変更点を記録しています。

## AIの使用

Gemini 4 Argonを教師実演の収集に使用し、採用した実演でSmolVLAを追加学習してロボット操作を推論しました。人間が実演の動画品質を確認し、学習に用いる実演を採用しました。
