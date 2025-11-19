import os
from django.shortcuts import render
from .ml_model.predict import predict_disease

def home(request):
    if request.method == "POST":
        image_file = request.FILES["image"]
        model_choice = request.POST.get("model_choice")

        # Save uploaded file temporarily
        upload_dir = "media/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, image_file.name)
        with open(file_path, "wb+") as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Run prediction
        label, probability = predict_disease(model_choice, file_path)

        context = {
            "uploaded_image": file_path,
            "prediction": label,
            "probability": f"{probability:.2f}",
            "model_choice": model_choice
        }
        return render(request, "home.html", context)

    return render(request, "home.html")
