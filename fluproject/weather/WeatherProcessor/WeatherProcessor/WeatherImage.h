#pragma once

// WeatherImage.h
// TODO: 
class WeatherImage {
private:
	Image e;
	int colorMap[];
	void downloadImage(char *url); // Crawls image
	void cropImage(bool isForecast);

public:
	WeatherImage(char *url, bool isForecast); // Crawls Image with For
	WeatherImage(); // Default Constructor
	int colorValue(int x, int y); // Gets color value from pixel location
	int match(int color); // Matc0h.m d +
	color from the color map


};