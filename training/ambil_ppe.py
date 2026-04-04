from roboflow import Roboflow

rf = Roboflow(api_key="I9mHYlk0TxPVnR21LdAl")

project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
version  = project.version(28)

print("Downloading model...")
version.download("yolov8", location="~/proyek-k3/ppe-dataset")
print("✅ Selesai!")
