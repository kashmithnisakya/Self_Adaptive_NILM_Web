// fetchDataAndDrawDeviceChart(1000, 2000, "fridge", 3);



function fetchDataAndDrawDeviceChart(
  startIndex,
  endIndex,
  device,
  year,
  targetAndPredictions
) {
  if (year <11) {
    const enteredDataSummmery = document.getElementById(
      "submited-data-container"
    );
    enteredDataSummmery.innerHTML = `<div class="submited-data"><p>Year <span class="energy-usage-value">${year}</span> data of &nbsp;<span class="energy-usage-value">${device}</span> &nbsp;from <span class="energy-usage-value">${startIndex}</span> to <span class="energy-usage-value">${endIndex}</span></p></div>`;
  }


    drawDeviceChart(targetAndPredictions, startIndex, device);

}

function calculateGraphArea(dataArray) {
  const traget_array = dataArray[0].map((element) => element * 6);
  const Predicted_array = dataArray[1].map((element) => element * 6);

  const tragetUsage =
    traget_array.reduce(
      (accumulator, currentValue) => accumulator + currentValue,
      0
    ) /
    (3600 * 1000);
  const PredictedUsage =
    Predicted_array.reduce(
      (accumulator, currentValue) => accumulator + currentValue,
      0
    ) /
    (3600 * 1000);

  const usageAccuracy =
    100 - Math.abs(((PredictedUsage - tragetUsage) / tragetUsage) * 100); //ACT-PRED/ACT

  return [
    tragetUsage.toFixed(4),
    PredictedUsage.toFixed(4),
    usageAccuracy.toFixed(2),
  ];
}

function mapDeviceDataToPlot(arrayOfData, startIndex) {
  if (
    Array.isArray(arrayOfData) &&
    arrayOfData.length > 0 &&
    Array.isArray(arrayOfData[0])
  ) {
    let rowsOfData = arrayOfData[0].map((value, index) => [
      startIndex + index,
      value,
      arrayOfData[1][index],
    ]);
    // console.log(rowsOfData);
    return rowsOfData;
  } else {
    console.log("Invalid input or empty array");
  }
}

google.charts.load("current", { packages: ["line"] });
// google.charts.setOnLoadCallback(drawDeviceChart);
function drawDeviceChart(dataToPlot, startIndex, device) {
  // console.log("dat to plot", dataToPlot);

  var data = new google.visualization.DataTable();

  data.addColumn("number", "Time (6s samples)");
  data.addColumn("number", "Target");
  data.addColumn("number", "Prediction");

  data.addRows(mapDeviceDataToPlot(dataToPlot, startIndex));

  var options = {
    // chart: {
    //   title: `Power consumption of ${device}`,
    // },
    legend: { position: "right", alignment: "center" },
    width: 1400,
    height: 400,
    axes: {
      x: {
        0: { side: "bottom" },
      },
    },
    colors: ["blue", "orange"],
    hAxis: {
      gridlines: { count: 5 }, // Adjust the count as per your requirement for horizontal gridlines
    },
    vAxis: {
      gridlines: { count: 5 }, // Adjust the count as per your requirement for vertical gridlines
      title: "Power (W)",
      slantedText: false,
    },
  };

  var chart = new google.charts.Line(document.getElementById("device-chart"));
  chart.draw(data, google.charts.Line.convertOptions(options));

  const [targetUsage, PredictedUsage, usageAccuracy] =
    calculateGraphArea(dataToPlot);
  console.log(targetUsage, PredictedUsage);
  const chartSummery = document.getElementById("device-chart-summery");
  const summetyContent = `<div class="chart-title">Power consumption of ${device}</div>
  <div class="energy-usage-container">
  <div class="energy-usage" >Target Energy Usage: <span class="energy-usage-value">${targetUsage}</span> kWh</div>
  <div class="energy-usage" >Predicted Energy Usage: <span class="energy-usage-value">${PredictedUsage}</span> kWh</div>
  <div class="energy-usage" >Accuracy: <span class="energy-usage-value">${usageAccuracy}</span> %</div>
  </div>`;
  chartSummery.innerHTML = summetyContent;
}

window.fetchDataAndDrawDeviceChart = fetchDataAndDrawDeviceChart;
