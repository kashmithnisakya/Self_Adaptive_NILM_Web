// google.charts.setOnLoadCallback(drawChart);

const deviceList = ["fridge"]; //add devices to this array

let formData = {
  year: 0,
  from: 0,
  to: 0,
};

function submitForm() {
  // Get form data
  formData = {
    year: document.getElementById("year").value,
    from: document.getElementById("from").value,
    to: document.getElementById("to").value,
  };

  // You can perform further operations here such as validation or AJAX submission

  // For demonstration, just log the form data
  // console.log(formData['from'],formData['to'],formData['device']);
  plotDeviceEnergyUsage(
    deviceList,
    parseInt(formData["from"]),
    parseInt(formData["to"]),
    parseInt(formData["year"])
  );
  // fetchDataAndDrawChart(1000, 5000, "fridge", 3);
}

function fetchDeviceData(url) {
  return fetch(url)
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json();
    })
    .catch((error) => {
      console.error("Error fetching data:", error);
    });
}

function plotDeviceEnergyUsage(deviceList, startIndex, endIndex, year) {

  if (year <11) {
    const enteredDataSummmery = document.getElementById(
      "submited-data-container"
    );
    enteredDataSummmery.innerHTML = `<div class="submited-data"><p>Year <span class="energy-usage-value">${year}</span> data &nbsp;from <span class="energy-usage-value">${startIndex}</span> to <span class="energy-usage-value">${endIndex}</span></p></div>`;
  }

  let fileUrls = [];
  for (let device of deviceList) {
    let newUrl = `../data/SelfAdaptive_prediction/${year}_th_year_${device}_prediction.json`;
    fileUrls.push(newUrl);
  }

  const aggregateUrl = `../data/aggregated_power/${year}_th_year_aggregated_power.json`;
  fileUrls.push(aggregateUrl);

  //console.log(fileUrls);

  let loadedDeviceData = [];
  //console.log(startIndex, endIndex, fileUrls);

  let fetchPromises = []; // Array to store promises for each fetch operation

  for (let url of fileUrls) {
    //console.log("hi0");
    // Push the promise returned by fetchData into the fetchPromises array
    fetchPromises.push(
      fetchDeviceData(url).then((fetchedData) => {
        // console.log(fetchedData);
        if (fetchedData.length === 3) {
          const slicedAggregate = fetchedData.map((element) =>
            element.slice(startIndex, endIndex)
          );
          // console.log(slicedAggregate);
          loadedDeviceData.push(slicedAggregate);
        } else {
          let slicedElements = fetchedData.slice(startIndex, endIndex);
          loadedDeviceData.push(slicedElements);
          //console.log("hi1");
        }
      })
    );
  }

  // Wait for all fetch operations to complete
  Promise.all(fetchPromises)
    .then(() => {
      //console.log("hi2");
      ///////////output is here
      //console.log("about to send");

      takeDataToOutSide(loadedDeviceData);
    })
    .catch((error) => {
      console.error("Error fetching data:", error);
    });

  // console.log(loadedDeviceData);
}

function takeDataToOutSide(dataObtained) {
  // console.log(dataObtained)
  //console.log(dataObtained);
  calculateEnergyUsage(dataObtained);
}

function calculateEnergyUsage(dataToFindArea) {
  //console.log(dataToFindArea);
  let lotalPowerValues = [];
  for (let data of dataToFindArea) {
    if (data.length === 3) {
      let phaseEnergy = 0;
      for (let i = 0; i < 3; i++) {
        // console.log('hear bitch')
        const phaseEnergyArray = data[i].map((element) => element * 6);
        const accumPhase =
          phaseEnergyArray.reduce(
            (accumulator, currentValue) => accumulator + currentValue,
            0
          ) /
          (3600 * 1000);
        // console.log(accumPhase)
        phaseEnergy += accumPhase;
      }
      // console.log('accumephse', phaseEnergy)
      lotalPowerValues.push(phaseEnergy);
    } else {
      const EnergyArray = data.map((element) => element * 6);
      const accumulatedEnergy =
        (EnergyArray.reduce(
          (accumulator, currentValue) => accumulator + currentValue,
          0
        ) /
          (3600 * 1000)) *
        3;

      lotalPowerValues.push(accumulatedEnergy);
    }
  }
  //console.log(lotalPowerValues);

  makeTable(lotalPowerValues, deviceList);
}

function makeTable(lotalPowerValues, deviceList) {
  //console.log(deviceList);
  const tableOfData = [["Device", "Energy Usage (kWh)"]];
  for (let i = 0; i < deviceList.length + 1; i++) {
    if (i === deviceList.length) {
      const totalOfDevices = lotalPowerValues.reduce(
        (accumulator, currentValue) => accumulator + currentValue,
        0
      );
      const datapoint = [
        "Other loads",
        parseInt(2 * lotalPowerValues[i] - totalOfDevices),
      ];
      tableOfData.push(datapoint);
    } else {
      const datapoint = [`${deviceList[i]}`, parseInt(lotalPowerValues[i])];
      tableOfData.push(datapoint);
    }
  }
  drawChart(tableOfData);
}

google.charts.load("current", { packages: ["corechart"] });
function drawChart(tableOfData) {
  //console.log(tableOfData);
  var data = google.visualization.arrayToDataTable(tableOfData);

  var options = {
    title: "Energy Consumption Comparison",
    legend: { position: 'right', alignment: 'center' }
  };

  var chart = new google.visualization.PieChart(
    document.getElementById("piechart")
  );

  chart.draw(data, options);
}
