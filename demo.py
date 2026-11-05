# -*- coding: utf-8 -*-
"""4char-captcha-onnx —— 最小推理示例

依赖: onnxruntime, numpy, pillow
用法: python demo.py 图片.png [更多图片...]
"""
import sys
import json
import os

import numpy as np
from PIL import Image
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "captcha_model.onnx")
CFG = os.path.join(HERE, "charset.json")


class Captcha:
    """4 字符验证码识别器（CTC 解码 + 4 头投票）"""

    def __init__(self, model_path=MODEL, cfg_path=CFG):
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        self.chars = cfg["charset"]      # index 0 是 CTC blank，字符从 1 开始
        self.height = int(cfg["height"])  # 固定 64
        self.nhead = int(cfg.get("heads", 4))

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(model_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        names = [o.name for o in self.sess.get_outputs()]
        # ens = 4 个头 log-softmax 之和，等权集成
        self.out_ens = "ens"
        self.out_heads = ["h%d" % i for i in range(self.nhead) if "h%d" % i in names]

    def _prep(self, im):
        """灰度 -> 等比缩放到高 64 -> [1,1,64,W] float32"""
        im = im.convert("L")
        w, h = im.size
        nw = max(8, int(round(w * self.height / float(h))))
        arr = np.asarray(im.resize((nw, self.height), Image.LANCZOS), dtype=np.float32) / 255.0
        return arr[None, None, :, :]

    def _ctc(self, ids):
        """贪心解码: 合并相邻重复 -> 去掉 blank

        注意 last 必须无条件更新, 否则 "5 blank 5" 这种会被吞掉一个字。
        """
        out, last = [], -1
        for i in ids:
            if i != last:
                if i != 0:
                    out.append(self.chars[i])
                last = i
        return "".join(out)

    def run(self, image):
        x = self._prep(image)
        want = [self.out_ens] + self.out_heads
        res = dict(zip(want, self.sess.run(want, {self.inp: x})))

        text = self._ctc(res[self.out_ens].argmax(axis=1).tolist())
        heads = [self._ctc(res[n].argmax(axis=1).tolist()) for n in self.out_heads]
        votes = sum(1 for t in heads if t == text)

        # 放行闸门: 必须正好 4 个字, 且 4 个头全部同意
        accepted = (len(text) == 4 and votes == len(self.out_heads))
        return {
            "text": text,
            "votes": "%d/%d" % (votes, len(self.out_heads)),
            "heads": heads,
            "accepted": accepted,
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cap = Captcha()
    for p in sys.argv[1:]:
        r = cap.run(Image.open(p))
        print("%-28s -> %-6s %s  %s" % (os.path.basename(p), r["text"], r["votes"],
                                        "ACCEPT" if r["accepted"] else "reject"))
