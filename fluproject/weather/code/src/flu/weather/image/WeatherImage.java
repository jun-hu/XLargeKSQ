package flu.weather.image;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;

import javax.imageio.ImageIO;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

public class WeatherImage {
	public static int TEMPERATURE=1;
	public static int PRECIPITATION=2;
	public static int HUMIDITY=3;
	public static int WIND=4;
	
	private BufferedImage image;
	public boolean read(String url){
		try {
			Document d=Jsoup.connect(url).get();
			String actualImage=d.getElementsByAttributeValue("name", "anim").get(0).attr("src");
			image=ImageIO.read(new URL("http://www.kma.go.kr"+actualImage));
			if(image==null){
				System.err.println("Cannot bring image");
				return false;
			}
//			for(double x=23.0;x<475;x+=14.42){
//				System.out.print(image.getRGB(464, (int)x)+",");
//			}
		} catch (IOException e) {
			System.err.println("IO Error");
			return false;
		}  catch (NullPointerException e) {
			System.err.println("Cannot bring image");
			return false;
		}
		return true;
	}
	public boolean readMask(boolean preMask){
		try {
			image=ImageIO.read(new File(preMask?"preMask.png":"postMask.png"));
			if(image==null){
				System.err.println("Cannot bring image");
				return false;
			}
		} catch (IOException e) {
			System.err.println("IO Error");
			return false;
		}  catch (NullPointerException e) {
			System.err.println("Cannot bring image");
			return false;
		}
		return true;
	}
	public double get(int x, int y, int param){
		switch(param){
		case 1:
			return ColorPicker.currentTemperature.match(image.getRGB(x, y));
		case 2:
			return ColorPicker.currentPrecipitation.match(image.getRGB(x, y));
		case 3:
			return ColorPicker.currentHumidity.match(image.getRGB(x, y));
		case 4:
			return ColorPicker.currentWind.match(image.getRGB(x, y));
		}
		return ColorPicker.COLORPICKER_NOMATCH;
	}
}
