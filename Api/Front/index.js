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
            console.log(data[key]);
            var p = document.createElement("p");
            p.classList = "mx-auto p-1 rounded-full text-center text-white font-bold w-full";
            if (data[key]['pred'] == 1){
                p.classList.add('bg-red-500')
                p.innerHTML = `${key} is toxic`;
            }else{
                p.classList.add('bg-green-500')
                p.innerHTML = `${key}  is not toxic`;
            }
            
            ul.appendChild(p);
        }
        console.log(data);
        predDiv.innerHTML = ul.innerHTML
      })
}

// document.querySelector("#selfieInput").addEventListener("change", askToxicity);
document.querySelector("#predict").addEventListener("click", askToxicity);

