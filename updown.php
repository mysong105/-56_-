<?php
$target_dir = "./camera/";
$target_file = $target_dir . basename($_FILES["fileToUpload"]["name"]);
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));

// Allow certain file formats
if($imageFileType != "jpg" && $imageFileType != "png" && $imageFileType != "jpeg"
&& $imageFileType != "gif" ) {
    echo "Sorry, only JPG, JPEG, PNG & GIF files are allowed.";
    $uploadOk = 0;
}

// Check if $uploadOk is set to 0 by an error
if ($uploadOk == 0) {
    echo "Sorry, your file was not uploaded.";
// if everything is ok, try to upload file
} else {
    if (move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $target_file)) {
		$filename = $_FILES["fileToUpload"]["name"];
		$imgurl = "http://203.255.176.79:13000/repos_camera". $_FILES["fileToUpload"]["name"];
		$size = $_FILES["fileToUpload"]["size"];

		$conn = mysqli_connect("localhost", "root", "cjsdpdnjs", "test");
		//images 테이블에 이미지정보를 저장하자.
		$sql = "insert into imagetest(filename, imgurl, size) values('$filename','$imgurl','$size')";
		mysqli_query($conn,$sql);
		mysqli_close($conn);

        	echo "<p>The file ". basename( $_FILES["fileToUpload"]["name"]). " has been uploaded.</p>";
		
		chdir("/home/user01/mytrain");
		exec("/home/user01/venv/bin/python servercrop.py /var/www/html/camera/".$filename);
		exec("/home/user01/venv/bin/python test.py");
		chdir("/home/user01/var/www/html");

		exec("mv /var/www/html/camera/".basename($_FILES["fileToUpload"]["name"])." /var/www/html/repos_camera");

		isfile = $_SERVER['DOCUMENT_ROOT']."/save_camera/".$filename."_1.jpg";
		clearstatcache();
		if(file_exists($isfile)){
			http_response_code(200);
		}else{
			http_response_code(404);
		}

		echo "<br><button type='button' onclick='history.back()'>돌아가기</button>";
    } else {
        echo "<p>Sorry, there was an error uploading your file.</p>";
		echo "<br><button type='button' onclick='history.back()'>돌아가기</button>";
    }
}
?>

