const API_URL = 'http://127.0.0.1:3000/predict';


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
            p.classList = "mx-auto p-1 m-1 w-1/2 rounded-full text-center text-white font-bold w-full";
            if (data[key]['pred'] == 1){
                p.classList.add('bg-red-500')
            }else{
                p.classList.add('bg-green-500')
            }
            p.innerHTML = data[key]['selfie'] +':'+ data[key]['pred'];
            ul.appendChild(p);
        }
        console.log(data);
        predDiv.innerHTML = ul.innerHTML
      })
}

document.querySelector("#selfieInput").addEventListener("change", askToxicity);
// document.querySelector("#predict").addEventListener("click", askToxicity);

