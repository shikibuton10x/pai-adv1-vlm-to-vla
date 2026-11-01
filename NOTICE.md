# 出典・改変・公開範囲

本パッケージは本人の講座課題用研究コードからの公開用コピーです。元の研究結果は変更していません。個人機械の絶対パスと本人private HF保管先を設定中のplaceholderへ置換し、認証/クラウド起動用スクリプトを除外しました。

- `safety.py`は既存本人の講座コードから保持、`env.py`は同環境アダプタを明示パス/履歴記録/有界resetへ拡張しました。
- 手先制御と表示方向は既存本人コードを参照し、今回の短操作/収録へ変更。対応は `docs/SOURCE_PROVENANCE.json`の元SHAとdestinationを参照。
- `vendor/measurement_stack.py`は公開LeRobotのfactory/processorを用いる測定ラッパーで、上流ライブラリ本体は同梱していません。
- 公開LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO （MIT）
- LeRobot / SmolVLA: https://github.com/huggingface/lerobot / https://huggingface.co/lerobot/smolvla_libero （各公開元のApache-2.0表示を参照）
- 本人の既存講座コードのライセンスはApache-2.0。第三者データ・重みの権利は元の提供者に帰属し、本packageのLICENSEでは再許諾しません。

大容量私有入力・raw API応答・認証情報・PARCの非公開提出物・チーム内会話は非収録です。小集計は記録結果の照合用であり、方策の性能を再走行で検証するものではありません。
