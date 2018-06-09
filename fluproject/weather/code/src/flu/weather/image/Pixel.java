package flu.weather.image;

public class Pixel {
	private double temperature,humidity,precipitation,wind;

	public Pixel(double temperature, double humidity, double precipitation, double wind) {
		super();
		this.temperature = temperature;
		this.humidity = humidity;
		this.precipitation = precipitation;
		this.wind = wind;
	}

	public double getTemperature() {
		return temperature;
	}

	public double getHumidity() {
		return humidity;
	}

	public double getPrecipitation() {
		return precipitation;
	}

	public double getWind() {
		return wind;
	}
	public boolean invalid(){
		return temperature==ColorPicker.COLORPICKER_NOMATCH
			&&humidity==ColorPicker.COLORPICKER_NOMATCH
			&&precipitation==ColorPicker.COLORPICKER_NOMATCH
			&&wind==ColorPicker.COLORPICKER_NOMATCH;
	}
	public void invalidate(){
		temperature=ColorPicker.COLORPICKER_NOMATCH;
		humidity=ColorPicker.COLORPICKER_NOMATCH;
		precipitation=ColorPicker.COLORPICKER_NOMATCH;
		wind=ColorPicker.COLORPICKER_NOMATCH;
	}
	private void add(Pixel p){
		temperature+=p.temperature;
		humidity+=p.humidity;
		precipitation+=p.precipitation;
		wind+=p.wind;
	}
	public Pixel(){
		invalidate();
	}
	public Pixel(Pixel p1,Pixel p2,Pixel p3,Pixel p4,Pixel p5,Pixel p6,Pixel p7,Pixel p8){
		int count=0;
		temperature=0;
		humidity=0;
		precipitation=0;
		wind=0;
		if(!p1.invalid()){ add(p1); count++; }
		if(!p2.invalid()){ add(p2); count++; }
		if(!p3.invalid()){ add(p3); count++; }
		if(!p4.invalid()){ add(p4); count++; }
		if(!p5.invalid()){ add(p5); count++; }
		if(!p6.invalid()){ add(p6); count++; }
		if(!p7.invalid()){ add(p7); count++; }
		if(!p8.invalid()){ add(p8); count++; }
		if(count!=0){
			temperature/=count;
			humidity/=count;
			precipitation/=count;
			wind/=count;
		}
	}

	@Override
	public String toString() {
		return "Pixel [temperature=" + temperature + ", humidity=" + humidity + ", precipitation=" + precipitation
				+ ", wind=" + wind + "]";
	}
	
}
