let formData = {
  device: 'none',
  from: 0,
  to: 0
};

async function submitForm(typeOfModel) {
  // Get form data
  formData = {
      device_name: document.getElementById("device").value,
      year: document.getElementById("year").value,
      start_point: document.getElementById("from").value,
      end_point: document.getElementById("to").value
  };
  
  // You can perform further operations here such as validation or AJAX submission
  
  // For demonstration, just log the form data
  console.log(formData);
  
  try {
      const res = await axios.post('http://localhost:8000/predict/', formData);
      setResponse(res.data); // Call to the function to handle the response
  } catch (error) {
      console.error('There was an error!', error);
  }
}

function setResponse(data) {
  // Handle the response data here

  // console.log("Response from backend:", data);
  // console.log(typeof data.response);
  const dataRecievedJason = data.response.replace(/'/g, '"');
  const dataRecieved = JSON.parse(dataRecievedJason);
  
  // console.log('iamhere'); 
  console.log(dataRecieved.disaggregated_power_prediction); 
  // console.log('iamhere'); 

  const target = dataRecieved.disaggregated_power_target;
  const prediction = dataRecieved.disaggregated_power_prediction;

  // console.log('iamh target and pred'); 
  const targetAndPredctionsArray= [target, prediction];
  // console.log(targetAndPredctionsArray);


  const agg_phase1 = dataRecieved.aggregated_power_phase_1;
  const agg_phase2 = dataRecieved.aggregated_power_phase_2;
  const agg_phase3 = dataRecieved.aggregated_power_phase_3;

  const aggregatedArray = [agg_phase1, agg_phase2, agg_phase3];

  fetchDataAndDrawAggregatedChart(parseInt(formData['start_point']),parseInt(formData['end_point']), parseInt(formData['year']),formData['device_name'],aggregatedArray);
  fetchDataAndDrawDeviceChart(parseInt(formData['start_point']),parseInt(formData['end_point']),formData['device_name'],parseInt(formData['year']),targetAndPredctionsArray);

}