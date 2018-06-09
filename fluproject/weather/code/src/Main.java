import flu.weather.image.ColorPicker;
import flu.weather.image.Pixel;
import flu.weather.image.WeatherImage;

public class Main {

	public static void main(String[] args) {
		// mhc: humidity
		// mWs: windspeed
		// mtc: temperature
		// mrh: rainfall
		// Image
		Pixel[][] p=new Pixel[455][455];
		//Pixel[][] p2=new Pixel[455][455];
		WeatherImage imgTemperature=new WeatherImage();
		WeatherImage imgPrecipitation=new WeatherImage();
		WeatherImage imgHumidity=new WeatherImage();
		WeatherImage imgWind=new WeatherImage();
		
		// Bring the images
		imgTemperature.read("http://www.kma.go.kr/cgi-bin/aws/nph-aws_ana1?201709140935_0_0_mtc_460_A0_CENN_a_15_1_h");
		imgPrecipitation.read("http://www.kma.go.kr/cgi-bin/aws/nph-aws_ana1?201709140935_0_0_mrh_460_A0_CENN_a_15_1_h");
		imgHumidity.read("http://www.kma.go.kr/cgi-bin/aws/nph-aws_ana1?201709140935_0_0_mhc_460_A0_CENN_a_15_1_h");
		imgWind.read("http://www.kma.go.kr/cgi-bin/aws/nph-aws_ana1?201709140935_0_0_mWs_460_A0_CENN_a_15_1_h");
		
		WeatherImage preMask=new WeatherImage();
		preMask.readMask(true);
		WeatherImage postMask=new WeatherImage();
		postMask.readMask(false);

		int x,y;
		// Raw Image
		for(x=0;x<455;x++)
			for(y=0;y<455;y++){
				p[x][y]=new Pixel(imgTemperature.get(x, y, WeatherImage.TEMPERATURE),
						imgHumidity.get(x, y, WeatherImage.HUMIDITY),
						imgPrecipitation.get(x, y, WeatherImage.PRECIPITATION),
						imgWind.get(x, y, WeatherImage.WIND));
			}
		
		System.out.println("Seoul:"+p[162][120]);
		System.out.println("Daejeon:"+p[189][205]);
		System.out.println("Daegu:"+p[267][253]);
		System.out.println("Busan:"+p[289][288]);
		System.out.println("Jeju:"+p[138][419]);
		
		/*
		// Filling the gap
		for(x=1;x<454;x++)
			for(y=1;y<454;y++){
				p[x][y]=new Pixel(imgTemperature.get(x, y, WeatherImage.TEMPERATURE),
						imgPrecipitation.get(x, y, WeatherImage.PRECIPITATION),
						imgHumidity.get(x, y, WeatherImage.HUMIDITY),
						imgWind.get(x, y, WeatherImage.WIND));
			}
		
		
		// Post-masking
		*/
	}

}
