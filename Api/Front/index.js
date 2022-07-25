const API_URL = 'https://3d4d-3-15-163-56.ngrok.io/predict';


function askToxicity(){
    var selfieInput = document.querySelector("#selfieInput").value;
    axios.post(API_URL, {
        'selfie':selfieInput
      })
      .then(function (response) {
        var ul = document.createElement("div");
        var predDiv = document.querySelector("#preds");
        var data = response.data
        for (var key in data){
            // console.log(data[key]);
            var p = document.createElement("h1");
            p.classList = "mx-auto p-1 rounded-lg text-center font-bold w-full";
            if (data[key]['pred'] == 1){
                p.innerHTML = `<p>${key} <strong class="p-1 rounded-lg bg-red-500 text-white">is toxic</strong></p>`;
            }else{
                p.innerHTML = `<p>${key} <strong class="p-1 rounded-lg bg-green-500 text-white">is not toxic</strong></p>`;
            }
            
            ul.appendChild(p);
        }
        // console.log(data);
        predDiv.innerHTML = ul.innerHTML
      })
}

// document.querySelector("#selfieInput").addEventListener("change", askToxicity);
document.querySelector("#predict").addEventListener("click", askToxicity);

