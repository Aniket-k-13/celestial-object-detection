
---
title: AstroDetect
emoji: 🔭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
license: mit
---

# 🔭 AstroDetect

## Deploy steps

```bash
# 1. Create Space: huggingface.co/new-space
#    SDK: Gradio | Hardware: CPU Basic (free)

# 2. Clone
git clone https://huggingface.co/spaces/YOUR_USERNAME/astrodetect
cd astrodetect

# 3. Copy files + add models
git lfs install
git lfs track "*.pt"
echo "*.pt filter=lfs diff=lfs merge=lfs -text" >> .gitattributes

cp app.py requirements.txt README.md .
cp resnet50_classifier.pt  model/   # from Google Drive/CelestialV2/
cp yolo11s_cosmica.pt      model/   # from Kaggle output tab
cp cosmica_classes.txt     model/   # from Kaggle output tab

# 4. Push
git add . && git commit -m "deploy" && git push
```
