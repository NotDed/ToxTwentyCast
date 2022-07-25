function toggleFile(){
    var inSelf = document.querySelector("#selfieInput")
    if (inSelf.type == "text"){
      inSelf.type = "File"
      document.querySelector("#toggler").textContent = "Individual prediction"
    }else{
      inSelf.type = "text"
      document.querySelector("#toggler").textContent = "Multi prediction"
    }
  }

document.querySelector("#toggler").addEventListener("click", toggleFile);