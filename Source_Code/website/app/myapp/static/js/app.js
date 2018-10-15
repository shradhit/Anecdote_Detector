// register.html
passit=0;
function validateEmail(email) {
    var re = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    return re.test(String(email).toLowerCase());
}
function validateform() {
    var x = document.getElementById("p1");
    var y = document.getElementById("p2");
    var e = document.getElementbyId("email");
    if (e.value == "" || x.value == "" || y.value == "") {
	return false;
    }
    else if (x.value != y.value) {
	return false;
    }
    return validateEmail(e.value);
}
function matchTwoPasswords() {
    var x = document.getElementById("p1");
    var y = document.getElementById("p2");
    var z = document.getElementById("pascheck");
    if(x.value == "") {
	passit = 0;
	z.innerHTML = "";
    }
    else if(x.value == y.value) {
	z.style.color = "green";
	z.innerHTML = "Correct &#x2714";
	passit = 1;
    }
    else {
	passit = 2;
	z.style.color = "red";
	z.innerHTML = "Incorrect &#x2716;";
    }
}

//home.html
function process1() {
    console.log("Process1 executed");
    var textarea1 = document.getElementById("textarea").value;
    var formData = new FormData();
    if (textarea1 === "") {
	console.log("In if")
	var file = document.getElementById("input_file");
	file = file.files[0];
	console.log(file);
	formData.append("file", file, file.name);
	formData.append("textarea1", textarea1);
	$.ajax({
	    url: "/process",
	    type: "POST",
	    data: formData,
	    processData: false,
	    contentType: false,
	    beforeSend: function() {
		document.getElementById("loading").setAttribute("style", "display:inline;margin-left:-60px;height:40px;margin-top:8px;");
	    },
	    complete: function() {
		document.getElementById("loading").setAttribute("style", "display:none");
	    },
	    success: function(data) {
		console.log(data);
		document.getElementById("textarea").innerHTML = data.input;
		document.getElementById("output1").innerHTML = data.output;
		alert("Done");
	    }
	});
    }
    else {
	console.log("In else")
	console.log(textarea1);
	formData = textarea1;
	$.ajax({
	    url: "/process",
	    type: "POST",
	    data: {
		"data":formData
	    },
	    beforeSend: function() {
		document.getElementById("loading").setAttribute("style", "display:inline;margin-left:-60px;height:40px;margin-top:8px;");
	    },
	    complete: function() {
		document.getElementById("loading").setAttribute("style", "display:none");
	    },
	    success: function(data) {
		console.log(data);
		document.getElementById("textarea").innerHTML = data.input;
		document.getElementById("output1").innerHTML = data.output;
		alert("Done");
	    }
	});
    }

}