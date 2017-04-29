import static org.junit.Assert.*;
import lejos.nxt.Motor;

import org.junit.Test;


public class BrickControllerTest {

	@Test
	public void testParseCommandStringThreeDigits() {
		String commandString = "*af200#";
		
		MotorCommand command = BrickController.parseCommandString(commandString);
		
		assert(command.getMotor() == Motor.A);
		assert(command.getSpeed() == 200);
	}
	
	@Test
	public void testParseCommandStringTwoDigits() {
		String commandString = "*af20#";
		
		MotorCommand command = BrickController.parseCommandString(commandString);
		
		assert(command.getMotor() == Motor.A);
		assert(command.getSpeed() == 20);
	}
	
	@Test
	public void testParseCommandStringOneDigits() {
		String commandString = "*af2#";
		
		MotorCommand command = BrickController.parseCommandString(commandString);
		
		assert(command.getMotor() == Motor.A);
		assert(command.getSpeed() == 2);
	}
	
	@Test
	public void testParseCommandStringThreeDigitsBackward() {
		String commandString = "*ab200#";
		
		MotorCommand command = BrickController.parseCommandString(commandString);
		
		assert(command.getMotor() == Motor.A);
		assert(command.getSpeed() == -200);
	}

}
