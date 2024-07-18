document.addEventListener('DOMContentLoaded', () => {
    const user_in = document.getElementById('user_in');
    const img_display = document.getElementById('chosen-img');

    user_in.addEventListener("change", () => {
        if (user_in.files.length > 0) {
            console.log("File selected: " + user_in.files[0].name);
            const file = user_in.files[0];
            const reader = new FileReader();

            reader.onload = function(e) {
                img_display.src = e.target.result;
            };

            reader.readAsDataURL(file);
        }
    });
    console.log("FINISHED THE JS")
});
