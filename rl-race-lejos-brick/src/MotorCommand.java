import lejos.nxt.Motor;
import lejos.nxt.NXTRegulatedMotor;


public class MotorCommand {
	private char motor;
	private char direction;
	private int speed;
	
	public MotorCommand(char motor, char direction, int speed) {
		this.motor = motor;
		this.direction = direction;
		this.speed = speed;
	}
	
	public NXTRegulatedMotor getMotor() {
		if(motor == 'a') {
			return Motor.A;
		} else if(motor == 'b') {
			return Motor.B;
		} else {
			return null;
		}
	}
	
	public int getSpeed() {
		if(direction == 'f') {
			return speed;
		} else if(direction == 'b') {
			return speed * -1;
		} else {
			return 0;
		}
	}
}
