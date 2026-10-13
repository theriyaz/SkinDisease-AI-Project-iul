const imageInput =
    document.getElementById("imageInput");

const chooseBtn =
    document.getElementById("chooseBtn");

const dropArea =
    document.getElementById("dropArea");

const previewContainer =
    document.getElementById("previewContainer");

const previewImage =
    document.getElementById("previewImage");

const analyzeBtn =
    document.getElementById("analyzeBtn");

const loading =
    document.getElementById("loading");

const result =
    document.getElementById("result");

const diseaseName =
    document.getElementById("diseaseName");

const className =
    document.getElementById("className");

const confidence =
    document.getElementById("confidence");

const progressBar =
    document.getElementById("progressBar");

const themeBtn =
    document.getElementById("themeBtn");


let selectedFile = null;


// ==========================================
// CHOOSE IMAGE
// ==========================================

chooseBtn.addEventListener(
    "click",
    () => {

        imageInput.click();

    }
);


// ==========================================
// IMAGE SELECT
// ==========================================

imageInput.addEventListener(
    "change",
    (event) => {

        const file =
            event.target.files[0];

        if (file) {

            showImage(file);

        }

    }
);


// ==========================================
// SHOW IMAGE
// ==========================================

function showImage(file) {

    if (!file.type.startsWith("image/")) {

        alert(
            "Please select an image."
        );

        return;

    }


    selectedFile = file;


    previewImage.src =
        URL.createObjectURL(file);


    previewContainer.classList.remove(
        "hidden"
    );


    result.classList.add(
        "hidden"
    );

}


// ==========================================
// ANALYZE
// ==========================================

analyzeBtn.addEventListener(
    "click",
    async () => {

        if (!selectedFile) {

            alert(
                "Please select an image first."
            );

            return;

        }


        loading.classList.remove(
            "hidden"
        );


        result.classList.add(
            "hidden"
        );


        analyzeBtn.disabled = true;


        const formData =
            new FormData();


        formData.append(
            "image",
            selectedFile
        );


        try {

            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (data.error) {

                throw new Error(
                    data.error
                );

            }


            // Disease

            diseaseName.textContent =
                data.disease;


            // Class

            className.textContent =
                "Class: " +
                data.class;


            // Confidence

            const score =
                Number(
                    data.confidence
                );


            confidence.textContent =
                score.toFixed(2) +
                "%";


            progressBar.style.width =
                score + "%";


            // ==================================
            // GRAD-CAM IMAGE
            // ==================================

            const explanation =
                document.querySelector(
                    ".gradcam-placeholder"
                );


            if (data.gradcam) {

                explanation.innerHTML = `

                    <div class="gradcam-result">

                        <img
                            src="${data.gradcam}"
                            alt="Grad-CAM visualization"
                        >

                        <p>
                            Highlighted regions show
                            areas that influenced the
                            AI prediction.
                        </p>

                    </div>

                `;

            }


            // Show result

            result.classList.remove(
                "hidden"
            );


            result.scrollIntoView({
                behavior: "smooth"
            });


        }

        catch (error) {

            alert(
                "Error: " +
                error.message
            );

        }


        finally {

            loading.classList.add(
                "hidden"
            );

            analyzeBtn.disabled =
                false;

        }

    }
);


// ==========================================
// DARK MODE
// ==========================================

themeBtn.addEventListener(
    "click",
    () => {

        document.body.classList.toggle(
            "dark"
        );


        if (
            document.body.classList.contains(
                "dark"
            )
        ) {

            themeBtn.textContent =
                "☀️";

        }

        else {

            themeBtn.textContent =
                "🌙";

        }

    }
);

// ==========================================
// BACK TO TOP
// ==========================================

const backToTop =
    document.getElementById("backToTop");


window.addEventListener(
    "scroll",
    () => {

        if (window.scrollY > 400) {

            backToTop.classList.add("show");

        } else {

            backToTop.classList.remove("show");

        }

    }
);


backToTop.addEventListener(
    "click",
    () => {

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    }
);