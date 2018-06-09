// WeatherProcessor.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"

using namespace cv;
using namespace std;
/*
int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}
	Mat image;
	image = imread(argv[1], IMREAD_COLOR); // Read the file
	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}
*/
size_t imgSave(void *ptr, size_t size, size_t buf, void* userdata)
{
	FILE* stream = (FILE*)userdata;
	if (!stream) {
		printf("!!! Stream Fail");
		return false;
	}

	return fwrite((FILE*)ptr, size, buf, stream);
}

bool download_png(char* url)
{
	FILE* fp;
	errno_t err=fopen_s(&fp,"out.png", "wb");
	if (err!=0) {
		printf("!!! File Fail");
		return false;
	}

	// Generating cURL settings
	CURL* curlCtx = curl_easy_init();
	curl_easy_setopt(curlCtx, CURLOPT_URL, url); // Specifying URL
	curl_easy_setopt(curlCtx, CURLOPT_WRITEDATA, fp); // Specifying writing file
	curl_easy_setopt(curlCtx, CURLOPT_WRITEFUNCTION, imgSave); // Specifying procedure to save image
	curl_easy_setopt(curlCtx, CURLOPT_FOLLOWLOCATION, 1); // Followup

	// Generate cURL Code
	CURLcode rc = curl_easy_perform(curlCtx);
	if (rc) {
		printf("!!! CURLcode Fail");
		return false;
	}

	// Access to remore URL
	long res_code = 0;
	curl_easy_getinfo(curlCtx, CURLINFO_RESPONSE_CODE, &res_code);
	if (!((res_code == 200 || res_code == 201) && rc != CURLE_ABORTED_BY_CALLBACK))
	{
		printf("!!! Response code: %d\n", res_code);
		return false;
	}

	// Clearing cURL settings
	printf("CLEAR");
	curl_easy_cleanup(curlCtx);
	fclose(fp);

	return true;
}

int main(int argc, char** argv)
{
	int a = 0;
	printf("%d%d\n",a++, a);
	if (!download_png("http://www.kma.go.kr/img/aws/aws_mtv_201709140400_460_A0_CENSN_40.png"))
	{
		printf("!! Failed to download file: %s\n", "http://www.kma.go.kr/img/aws/aws_mtv_201709140400_460_A0_CENSN_40.png");
		return -1;
	}

	return 0;
}