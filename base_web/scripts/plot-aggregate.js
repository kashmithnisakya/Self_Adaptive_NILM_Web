
function fetchDataAndDrawAggregatedChart(startIndex, endIndex, year,device,phasesOfAggregated) {
 
  drawAggregateChart(phasesOfAggregated, startIndex, device);
 
}


function mapAggregateDataToPlot(arrayOfData, startIndex) {
  if (
    Array.isArray(arrayOfData) &&
    arrayOfData.length > 0 &&
    Array.isArray(arrayOfData[0])
  ) {
    let rowsOfData = arrayOfData[0].map((value, index) => [
      startIndex + index,
      value,
      arrayOfData[1][index],
      arrayOfData[2][index]
    ]);
    // console.log(rowsOfData);
    return rowsOfData;
  } else {
    console.log("Invalid input or empty array");
  }
}




google.charts.load("current", { packages: ["line"] });
// google.charts.setOnLoadCallback(drawAggregateChart);

function drawAggregateChart(dataToPlot, startIndex, device) {

  var dataAgg = new google.visualization.DataTable();
  dataAgg.addColumn("number", "Time (6s samples)");
  dataAgg.addColumn("number", "Phase-1");
  dataAgg.addColumn("number", "Phase-2");
  dataAgg.addColumn("number", "Phase-3");

  dataAgg.addRows(mapAggregateDataToPlot(dataToPlot, startIndex));

  var options = {
    // chart: {
    //   title: `Aggregated power`,
    // },
    legend: { position: 'right', alignment: 'center' },
    width: 1400,
    height: 400,
    axes: {
      x: {
        0: { side: "bottom" },
      },
    },
    colors: ["red","green", "orange"],
    hAxis: {
      gridlines: { count: 5 }, // Adjust the count as per your requirement for horizontal gridlines
    },
    vAxis: {
      gridlines: { count: 5 }, // Adjust the count as per your requirement for vertical gridlines
      title: "Power (W)",
      slantedText: false,
    },
  };

  var chartAgg = new google.charts.Line(
    document.getElementById("aggregate-chart")
  );
  chartAgg.draw(dataAgg, google.charts.Line.convertOptions(options));


  const chartSummery = document.getElementById("aggregate-chart-summery");
  const summetyContent = `<div class="chart-title">Aggregated power</div>`;

  chartSummery.innerHTML = summetyContent;


}


window.fetchDataAndDrawAggregatedChart = fetchDataAndDrawAggregatedChart;

