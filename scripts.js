let model_five_diseases = null;
let model_pneumonia = null;
let model_lung_cancer = null;
//let model_nature = null;
//#################################################################################
// ### MODELS LOAD
//#################################################################################
async function loadmodelsFunction(){
	model_five_diseases = await tf.loadLayersModel('https://raw.githubusercontent.com/bachtses/Medical-Report-Generation-Tool-First-Prototype/main/models/X-Ray_5_Diseases_Classification/model.json');
	model_pneumonia = await tf.loadLayersModel('https://raw.githubusercontent.com/bachtses/Medical-Report-Generation-Tool-First-Prototype/main/models/X-Ray_Pneumonia_Detection/model.json');
	model_lung_cancer = await tf.loadLayersModel('https://raw.githubusercontent.com/bachtses/Medical-Report-Generation-Tool-First-Prototype/main/models/X-Ray_Lung_Cancer_Classification/model.json');
   //model_nature = await tf.loadLayersModel('https://raw.githubusercontent.com/bachtses/Chest_X-Ray_Medical_Report_Web_App/main/models/X-Ray_Nature_Classification/model.json');
    console.log("MODEL LOADED!: Five Diseases model");
    console.log("MODEL LOADED!: Pneumonia model");
    console.log("MODEL LOADED!: Lung Cancer model");
	//console.log("MODEL LOADED!: Nature");
	
	await progressBar();
	

}


//#################################################################################
// ### DROPDOWN MENU
//#################################################################################
async function dropdownSelection() {
	$(".modality_option_container").fadeOut("fast");
	document.getElementById("right_container").style.display = "inline";
	$("#summary_title").hide();
	$("#summary_title").replaceWith(`<p id="summary_title">Notifications Panel</p>`);
	$('#summary_title').fadeIn(500);

	await loadmodelsFunction();

}


//#################################################################################
// ### DISPLAY DICOM IMAGE
//#################################################################################
cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
let loaded = false;

var element = document.getElementById('idImage');
cornerstone.enable(element);


//#################################################################################
// ### IMAGE LOAD
//#################################################################################
var allowed_extensions_for_dicom_formats = new Array("dcm");
var allowed_extensions_for_other_formats = new Array("jpeg","jpg","png");

async function showFiles() {
	// read the file from the user
    let file = document.querySelector('input[type=file]').files[0];
	console.log("\n\n\n\n\n\n\NEW UPLOADED FILE: \n\n", file)
	var fileName = file.name;
	var file_extension = fileName.split('.').pop().toLowerCase(); 
 

	if (allowed_extensions_for_dicom_formats.includes(file_extension)){
		$("#initial_image_display").fadeOut("fast");
		$(".cornerstone-canvas").fadeIn("fast");
		
		var imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(file);
		const element = document.getElementById('idImage');

		cornerstone.loadImage(imageId).then(function(image) {
			const viewport = cornerstone.getDefaultViewportForImage(element, image);
			cornerstone.displayImage(element, image, viewport);
		});

		setTimeout(() => {predict()}, 500);
		document.getElementById("divModelDownloadFraction").style.marginTop = "22%";
		document.getElementById("divModelDownloadFraction").innerHTML = "Image Processing...";
		
		
	}else if (allowed_extensions_for_other_formats.includes(file_extension)){
		$(".cornerstone-canvas").fadeOut("fast");
		$("#initial_image_display").fadeIn("fast");

		// An empty img element
		let demoImage = document.getElementById('initial_image_display');
		console.log("demoImage", demoImage)
		// read the file from the user
		let file = document.querySelector('input[type=file]').files[0];
		
		const reader = new FileReader();
		reader.onload = function (event) {
			demoImage.src = reader.result;     
		}
		reader.readAsDataURL(file);

		setTimeout(() => {predict()}, 500);
		document.getElementById("divModelDownloadFraction").style.marginTop = "22%";
		document.getElementById("divModelDownloadFraction").innerHTML = "Image Processing...";	
	}
}  
 

//#################################################################################
// ### PREDICTIONS
//#################################################################################
async function predict(){

	let file = document.querySelector('input[type=file]').files[0];
	var fileName = file.name;
	var file_extension = fileName.split('.').pop().toLowerCase(); 	
	if (allowed_extensions_for_dicom_formats.includes(file_extension)){
		var img_ = document.getElementById('idImage').getElementsByClassName("cornerstone-canvas")[0];
	}else if(allowed_extensions_for_other_formats.includes(file_extension)){
		var img_ = document.getElementById('initial_image_display');
	}

	console.log("img_", img_);

	//document.getElementById('testCanvas').getContext('2d').canvas.width = img_.width;
	//document.getElementById('testCanvas').getContext('2d').canvas.height = img_.height;
	//document.getElementById('testCanvas').getContext('2d').drawImage(img_, 0, 0, img_.width, img_.height);

	// Pre-process the image
	// model five diseases classification
	let tensor_5_diseases = tf.browser.fromPixels(img_)
	.resizeNearestNeighbor([224,224]) // change the image size here
	.toFloat()
	.div(tf.scalar(255.0))
	.expandDims();
	// model pneumonia classification
	let tensor_pneumonia = tf.browser.fromPixels(img_)
	.resizeNearestNeighbor([200,200]) // change the image size here
	.toFloat()
	.expandDims();
	// model lungcancer classification
	let tensor_lungcancer = tf.browser.fromPixels(img_)
	.resizeNearestNeighbor([128,128]) // change the image size here
	.toFloat()
	.div(tf.scalar(255.0))
	.expandDims();

	// model nature classification ---------->  PYTORCH MODEL
	//let tensor_nature = tf.browser.fromPixels(img_)
	//.resizeNearestNeighbor([150,150]) // change the image size here
	//.toFloat()
	//.div(tf.scalar(255.0))
	//.expandDims();
	//tensor_nature.shape.reverse();
	//tensor_nature = tf.squeeze(tensor_nature);
	//tensor_nature = tf.expandDims(tensor_nature);


	console.log("\n")
	console.log("inputs for each model:")
	console.log(tensor_5_diseases.shape)
	console.log(tensor_pneumonia.shape)
	console.log(tensor_lungcancer.shape)
	//console.log(tensor_nature.shape)

	document.getElementById("divModelDownloadFraction").style.display = "none";
	document.getElementById("divModelDownloadFraction").style.lineHeight = "0px";
	document.getElementById("divModelDownloadFraction").style.height = "0px";
	console.log("\n")


	let predictions_five_diseases = await model_five_diseases.predict(tensor_5_diseases).data(); //model disease predictions
	
	let predictions_pneumonia = await model_pneumonia.predict(tensor_pneumonia).data(); //model pneumonia predictions
	predictions_pneumonia[1] = 1 - predictions_pneumonia[0];
	
	let predictions_lungcancer = await model_lung_cancer.predict(tensor_lungcancer).data(); //model lungcancer predictions
	predictions_lungcancer[1] = 1 - predictions_lungcancer[0]; 

	//let predictions_nature = await model_nature.predict(tensor_nature).data(); //model nature classification


	var LABELS_five_diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'];
	var LABELS_pneumonia = ['Pneumonia', 'No Pneumonia'];
	var LABELS_lungcancer = ['Benign', 'Malignant'];

	let RESULTS_five_diseases = Array.from(predictions_five_diseases)  //i.e. const RESULTS = ["0", "0", "1", "1", "1"]
	RESULTS_five_diseases = RESULTS_five_diseases.map(function(each_element){
		return Number(each_element.toFixed(3));
	});

	let RESULTS_pneumonia = Array.from(predictions_pneumonia)
	RESULTS_pneumonia = RESULTS_pneumonia.map(function(each_element){
		return Number(each_element.toFixed(3));
	});

	let RESULTS_lungcancer = Array.from(predictions_lungcancer)
	RESULTS_lungcancer = RESULTS_lungcancer.map(function(each_element){
		return Number(each_element.toFixed(3));
	});

	console.log("predictions five diseases:\n"+ RESULTS_five_diseases)
	console.log("predictions pneumonia:\n"+ RESULTS_pneumonia)
	console.log("predictions lung cancer:\n"+ RESULTS_lungcancer)
	//console.log("predictions nature:\n"+ predictions_nature)

	
	// Construct the pre-generated templates
	var text = '';
	var found_a_disease = 0;
	var predefined_sentences = [['There are no signs of atelectasis. ', 'There are signs of atelectasis. '],
	['The heart size is within normal limits and cardiac silhouette is normal. ', 'The cardiac silhouette is enlarged. '],
	['No focal consolidation. ', 'There is focal consolidation. '],
	['No typical findings of pulmonary edema. ', 'Pulmonary edema is seen. '],
	['No pleural effusion. ', 'Pleural effusion is seen. ']];
	for (let i = 0; i < LABELS_five_diseases.length; i++) {
		if (RESULTS_five_diseases[i] >= 0.5) {
			text += predefined_sentences[i][1];
			found_a_disease = 1;
		}
		else {
			text += predefined_sentences[i][0];
		}
	}

	var text_pneumonia = '';
	if (predictions_pneumonia[0] > predictions_pneumonia[1]) {
		text_pneumonia = 'a probability of pneumonia'
		found_a_disease = 1;
	}
	else {
		text_pneumonia = 'no high probability of pneumonia'
	}

	var text_lungcancer = '';
	if (predictions_lungcancer[0] > predictions_lungcancer[1]) {
		text_lungcancer = 'benign lung tumor without any abnormal mass or nodule.'
	}
	else {
		text_lungcancer = 'malignant lung tumor with an abnormal mass or nodule.'
		found_a_disease = 1;
	}

	var text_clear_report = '';
	if (found_a_disease == 0) {
		text_clear_report = 'Overall, there is no evidence of active disease and the lungs are clear.'
	}

	var slow_velocity = 500;

	// Clear the previous session and display the results of medical report
	$('#summary_title').fadeOut(slow_velocity, function(){
		$("#summary_title").hide();
		$("#summary_title").replaceWith(`<p id="summary_title">Medical Report Summary</p>`);
		$('#summary_title').fadeIn(slow_velocity);
	}); 

	$('#image_name').fadeOut(slow_velocity, function(){
		$("#image_name").hide();
		$("#image_name").replaceWith(`<h1 id="image_name">Uploaded Image: imagename </h1>`);
		$('#image_name').fadeIn(slow_velocity);
	});
	$('#image_modality').fadeOut(slow_velocity, function(){
		$("#image_modality").hide();
		var todayDate = new Date().toISOString().slice(0, 10);
		$("#image_modality").replaceWith(`<h2>Imaging Modality: X-Ray &nbsp; Location: Chest &nbsp; Date: ${todayDate}</h2>`);
		$('#image_modality').fadeIn(slow_velocity);
	});
	$('#findings').fadeOut(slow_velocity, function(){
		$("#findings").hide();
		$("#findings").replaceWith(`<h3 id="findings" style="border-bottom: 1px solid #363634;">Findings</h3>`);
		$('#findings').fadeIn(slow_velocity);
	});
	$('#gridrow1').fadeOut(slow_velocity, function(){
		$("#gridrow1").hide();
		$("#gridrow1").replaceWith(`<div class="gridrow" id="gridrow1" style="border-bottom: 1px solid #363634;"><li>${LABELS_lungcancer[0]}</li><li>${LABELS_lungcancer[1]}</li></div>`);
		$('#gridrow1').fadeIn(slow_velocity);
	});
	$('#gridrow2').fadeOut(slow_velocity, function(){
		$("#gridrow2").hide();
		$("#gridrow2").replaceWith(`<div class="gridrow" id="gridrow2"><li>${RESULTS_lungcancer[0]}</li><li>${RESULTS_lungcancer[1]}</li></div>`);
		$('#gridrow2').fadeIn(slow_velocity);
	});
	$('#gridrow3').fadeOut(slow_velocity, function(){
		$("#gridrow3").hide();
		$("#gridrow3").replaceWith(`<div class="gridrow" id="gridrow3" style="border-bottom: 1px solid #363634;"><li>${LABELS_five_diseases[0]}</li><li>${LABELS_five_diseases[1]}</li><li>${LABELS_five_diseases[2]}</li><li>${LABELS_five_diseases[3]}</li><li>${LABELS_five_diseases[4]}</li></div>`);
		$('#gridrow3').fadeIn(slow_velocity);
	});
	$('#gridrow4').fadeOut(slow_velocity, function(){
		$("#gridrow4").hide();
		$("#gridrow4").replaceWith(`<div class="gridrow" id="gridrow4"><li>${RESULTS_five_diseases[0]}</li><li>${RESULTS_five_diseases[1]}</li><li>${RESULTS_five_diseases[2]}</li><li>${RESULTS_five_diseases[3]}</li><li>${RESULTS_five_diseases[4]}</li></div>`);
		$('#gridrow4').fadeIn(slow_velocity);
	});
	$('#gridrow5').fadeOut(slow_velocity, function(){
		$("#gridrow5").hide();
		$("#gridrow5").replaceWith(`<div class="gridrow" id="gridrow5" style="border-bottom: 1px solid #363634;"><li>${LABELS_pneumonia[0]}</li><li>${LABELS_pneumonia[1]}</li></div>`);
		$('#gridrow5').fadeIn(slow_velocity);
	});
	$('#gridrow6').fadeOut(slow_velocity, function(){
		$("#gridrow6").hide();
		$("#gridrow6").replaceWith(`<div class="gridrow" id="gridrow6"><li>${RESULTS_pneumonia[0]}</li><li>${RESULTS_pneumonia[1]}</li></div>`);
		$('#gridrow6').fadeIn(slow_velocity);
	});

	$('#prediction_list').fadeOut(slow_velocity, function(){
		$("#prediction_list").hide();
		$("#prediction_list").replaceWith(`<h4 id='prediction_list'>The patient of the x-ray image image name, has been diagnosed with ${text_lungcancer} ${text} Also, there is ${text_pneumonia} suggestion. ${text_clear_report}</h4>`);
		$('#prediction_list').fadeIn(slow_velocity);
	});

	//export button visible
	$('.export_button').fadeIn(slow_velocity);
	//text editable button visible
	$('.textedit_button').fadeIn(slow_velocity);

}


//#################################################################################
// ### LOADING PROGRESS BAR
//#################################################################################
function progressBar() {
	// loading progress bar
	var i = 0;
	if (i == 0) {
		i = 1;
		var elem = document.getElementById("progress_bar");
		var width = 1;
		var id = setInterval(frame, 10);
		function frame() {
			if (width >= 100) {
				clearInterval(id);
				i = 0;
				document.getElementById("divModelDownloadFraction").style.marginTop = "22%";
				document.getElementById("divModelDownloadFraction").innerHTML = "<b>Model Loaded Succesfully:</b> Five Diseases model <br><b>Model Loaded Succesfully:</b> Pneumonia model  <br><b>Model Loaded Succesfully:</b> Lung Cancer model<br><h1 style='font-size:20px;'>Please proceed to the image upload</h1>";
				document.getElementById("files_upload").disabled = false;
				document.getElementById("left_container").style.opacity = "1";
			} else {
				width++;
				elem.style.width = width + "%";
			}
		}
	}
}


//#################################################################################
// ### MAKE REPORT TEXT EDITABLE
//#################################################################################
async function MakeTextEditable(){
	var post = $("#prediction_list").clone();
	post.find("span:not(.post_tag):not(.post_mentioned)").remove();
	post = $.trim(post.text());
	console.log("Manual report edit")
	console.log("Characters count: ", post.length);
	//make the text editable
	$('#prediction_list').attr('contenteditable', 'true');
	//change the cursor style
	document.getElementById("prediction_list").style.cursor = "text";

	input = document.querySelector('#prediction_list');
    settings = {
      	maxLen: 430,
    }
	keys = {'backspace': 8,'shift': 16,'ctrl': 17,'alt': 18,'delete': 46,'leftArrow': 37,'upArrow': 38,'rightArrow': 39,'downArrow': 40, }
	utils = {special: {},navigational: {},
			isSpecial(e) {
		  		return typeof this.special[e.keyCode] !== 'undefined';
			},
			isNavigational(e) {
		  	return typeof this.navigational[e.keyCode] !== 'undefined';
			}
	}
	utils.special[keys['backspace']] = true;
	utils.special[keys['shift']] = true;
	utils.special[keys['ctrl']] = true;
	utils.special[keys['alt']] = true;
	utils.special[keys['delete']] = true;
	utils.navigational[keys['upArrow']] = true;
	utils.navigational[keys['downArrow']] = true;
	utils.navigational[keys['leftArrow']] = true;
	utils.navigational[keys['rightArrow']] = true;
	
	input.addEventListener('keydown', function(event) {
		let len = event.target.innerText.trim().length;
		hasSelection = false;
		selection = window.getSelection();
		isSpecial = utils.isSpecial(event);
		isNavigational = utils.isNavigational(event);
		if (selection) {
		  hasSelection = !!selection.toString();
		}
		if (isSpecial || isNavigational) {
		  return true;
		}
		if (len >= settings.maxLen && !hasSelection) {
		  event.preventDefault();
		  return false;
		}
	});
}


//#################################################################################
// ### EXPORT TO PDF
//#################################################################################
function CreatePDFfromHTML() {	
	let iframe = document.createElement("iframe");
	iframe.setAttribute("id", "iframe");
	iframe.style.width = "810px";
	iframe.style.height = "100%";
	iframe.style.visibility = 'hidden';      

	document.body.appendChild(iframe);
	let html = document.documentElement.outerHTML;
	iframe.srcdoc = html;
	iframe.addEventListener("load", () => {
		html2canvas(iframe.contentWindow.document.getElementById("results"),{background: '#fff', scale: 5}).then(function(canvas) {
			//let link = document.createElement("a");
			//link.download = "Medical_Report.jpg";
			//canvas.toBlob( function(blob) {
			//link.href = URL.createObjectURL(blob);
			//link.click();
			//}, 'image/jpg');
			//document.body.appendChild(canvas);
			var imgData = canvas.toDataURL("image/png", 1.0);
			var pdf = new jsPDF("p", "mm", "a4");
			pdf.addImage(imgData, 'PNG', 0, 40);
			pdf.save('report_export.pdf');
			$("#iframe").replaceWith(``);
		});
	});
	
	console.log("PDF exported")
}









