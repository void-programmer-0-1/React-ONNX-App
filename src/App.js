
import './App.css';
import React, { useState, useEffect } from 'react';
import { InferenceSession,Tensor  } from 'onnxruntime-web';

async function linearRegression(number) {
    try{
        const session = await InferenceSession.create("./onnx_model.onnx",{ executionProviders: ['webgl'] });
        const data = Float32Array.from([number]);
        const tensor_data = new Tensor('float32', data, [1, 1]);
        const feeds = { "input": tensor_data };
        const results = await session.run(feeds);
        return results;
    }
    catch (error) {
        return `failed to inference ONNX model: ${error}.`;
    }
}

function square(number){
	return number * 2;
}


function App() {
	
	const [userinput, setUserinput] = useState('0');
	const [prediction,setPrediction] = useState('0');

	useEffect(() => {
		if(userinput != "0"){
			linearRegression(parseInt(userinput)).then(result => { 
				result = Math.floor(result.output.data[0]).toString();
				setPrediction(result);
			}); 
		}
		else{
			setPrediction("0");
		}
	},[userinput]);


	return (

	<div className="App">

		<div className="card">
			<div className="card-container">
				<div className="app-intro">
					<p>Hey,Buddy I am AI,Enter a number to find it's square</p>
				</div>
			</div>
		</div>

		<div className="card">
			<div className="card-container">
				<div className="card-title">
					<p>Linear-Regression</p>
				</div>
				<div className="card-input">
					<label htmlFor="user-input">Enter a Number</label>
					<input onChange={event => { event.target.value != "" ? setUserinput(event.target.value) : setUserinput("0") } } id="user-input" type="number"></input>
				</div>
				<div className="card-submit">
					<p>{ prediction == "NaN" ? setPrediction("0") : prediction }</p>
				</div>
			</div>
		</div>

		<div className="card">
			<div className="card-container">
				<div className="app-intro">
					<p>{prediction == square(parseInt(userinput)) ? "correct" : `error  ${square(parseInt(userinput).toString())} is the correct value`  }</p>
				</div>
			</div>
		</div>

		
	</div>
	);
}

export default App;
