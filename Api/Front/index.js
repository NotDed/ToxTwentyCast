const API_URL = 'https://2efd-3-142-82-205.ngrok.io/predict';

const tableStyles = "border-collapse table-auto w-full";
const cellStyles = "mx-auto my-2 rounded-lg text-center font-bold w-full"
const headerStyles = "border-b font-bold text-center text-slate-900";
const cellElemetsStyle = "border-b border-slate-200 text-center text-slate-600 rounded-lg my-4"

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
    headerCell.appendChild(trHeaderCell);

    table.appendChild(headerCell);

    return table;

}

function createPredictionCell(key, data){
    var cell = document.createElement("tr");
    cell.classList = cellStyles

    var keyCell = document.createElement("td");
    var predCell = document.createElement("td");
    var trCell = document.createElement("td");

    keyCell.innerHTML = key;
    keyCell.classList = cellElemetsStyle
    keyCell.classList.toggle('text-left')

    predCell.innerHTML = data[key]['pred'] == 18 ? "Toxic" : "Non toxic";
    predCell.classList = cellElemetsStyle
    predCell.classList.add(data[key]['pred'] == 18 ? "bg-red-500" : "bg-green-500")

    trCell.innerHTML =  data[key]['with threshold'];
    trCell.classList = cellElemetsStyle

    cell.appendChild(keyCell);
    cell.appendChild(predCell);
    cell.appendChild(trCell);

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

function askToxicity(){
    var selfieInput = document.querySelector("#selfieInput").value.split(",");
    console.log(selfieInput)
    axios.post(API_URL, {
        'selfies':selfieInput
      })
      .then(function (response) {
        var predDiv = document.querySelector("#preds");
        var data = response.data;
        drawPredictions(data, predDiv)
      })
}

// document.querySelector("#selfieInput").addEventListener("change", askToxicity);
document.querySelector("#predict").addEventListener("click", askToxicity);

