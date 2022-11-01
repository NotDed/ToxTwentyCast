const API_URL = "https://3541-18-218-3-115.ngrok.io/predict";

const tableStyles = "border-collapse table-auto w-full";
const cellStyles = "mx-auto my-2 rounded-lg text-center font-bold w-full";
const headerStyles = "border-b font-bold text-center text-slate-900";
const cellElemetsStyle = "border-b border-slate-200 text-center text-slate-600 rounded-lg";
const errorStyle = "mx-auto my-2 w-full text-center font-bold rounded-lg text-white";

function createPredictionTable(){

    var table = document.createElement("table");
    table.classList = tableStyles

    var headerCell = document.createElement("tr");

    var keyHeaderCell = document.createElement("td");
    var predHeaderCell = document.createElement("td");
    var trHeaderCell = document.createElement("td");


    keyHeaderCell.classList = headerStyles;
    predHeaderCell.classList = headerStyles;
    trHeaderCell.classList = headerStyles;

    keyHeaderCell.innerHTML = "Selfie";
    predHeaderCell.innerHTML = "Predcition";
    trHeaderCell.innerHTML = "Threshold";

    headerCell.appendChild(keyHeaderCell);
    headerCell.appendChild(predHeaderCell);
    // headerCell.appendChild(trHeaderCell);

    table.appendChild(headerCell);

    return table;
}

function createPredictionCell(key, data){
    var cell = document.createElement("tr");
    cell.classList = cellStyles;

    var keyCell = document.createElement("td");
    var predCell = document.createElement("td");
    // var trCell = document.createElement("td");

    keyCell.innerHTML = key;
    keyCell.classList = cellElemetsStyle;
    keyCell.classList.replace('text-center','text-left');

    predCell.innerHTML = data[key]['pred'] ;
    predCell.classList = cellElemetsStyle;
    predCell.classList.replace('text-slate-600','text-white');
    predCell.classList.add(data[key]['pred'] > data[key]['with threshold'] ? "bg-red-500" : "bg-green-500");

    // trCell.innerHTML =  data[key]['with threshold'];
    // trCell.classList = cellElemetsStyle;

    cell.appendChild(keyCell);
    cell.appendChild(predCell);
    // cell.appendChild(trCell);

    return cell;
}

function drawPredictions(data, destinationElement){
    var predTable = createPredictionTable();
    for (var key in data){
        var predCell = createPredictionCell(key, data);
        predTable.appendChild(predCell);
    }
    destinationElement.innerHTML = predTable.outerHTML;
}

function drawError(erroValue, destinationElement){
    var errorPrompt = document.createElement("p");
    errorPrompt.classList = errorStyle;
    errorPrompt.innerHTML = erroValue;
    destinationElement.innerHTML = errorPrompt.innerHTML;
}

function askToxicity(){
    var selfieInput = document.querySelector("#selfieInput").value.split(",");
    var predOutput = document.querySelector("#preds");
    const options = {
        method: 'POST',
        url: API_URL,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': '*',
          'Access-Control-Allow-Credentials': 'true'
        },
        data: {
          selfie: selfieInput
        }
      };
      
      axios.request(options).then(function (response) {
        console.log(response.data);
        var data = response.data;
        drawPredictions(data, predOutput);
      }).catch(function (error) {
        console.error(error);
        drawError("something went wrong", predOutput);
      });

    // var options = {
    //     method: 'POST',
    //     url: API_URL,
    //     headers: {
    //       'Access-Control-Allow-Origin': '*',
    //       'Access-Control-Allow-Headers': '*',
    //       'Access-Control-Allow-Credentials': 'true'
    //     },
    //     data: {
    //       selfie: selfieInput
    //     }
    //   };

    // axios.post(API_URL, options)
    //   .then(function (response) {
    //     console.log(response.data);
    //     var data = response.data;
    //     drawPredictions(data, predOutput);
    //   }).catch(function (error) {
    //     drawError("something went wrong", predOutput);
    //   });
}

document.querySelector("#selfieInput").addEventListener("input", askToxicity);
document.querySelector("#predict").addEventListener("click", askToxicity);
