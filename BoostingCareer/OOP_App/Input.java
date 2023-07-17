import java.util.Scanner;

public class Input {
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		double r = sc.nextDouble();
		double pi = 3.14;
		System.out.println("Circumference = " + (2 * pi * r));
	}
}