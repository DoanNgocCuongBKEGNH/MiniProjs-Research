package game.gui;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class Drawer implements MouseListener, MouseMotionListener{
	protected Graphics myG;
	protected JPanel myPa;
	protected int width;
	protected int height;
	protected int mouseX;
	protected int mouseY;
	protected boolean mouseClicked = false;
	
	protected Drawer(JPanel myPa) {
		this.myPa = myPa;
		this.myPa.addMouseListener(this);
		this.myPa.addMouseMotionListener(this);
	}
	
	public void mousePressed() {
		if(mouseClicked) {
			return;
		}
		mouseClicked = false;
	}
	
	public void setMyG(Graphics g) {
		this.myG = g;
		Rectangle r = g.getClipBounds();
		this.height = (int)r.getHeight();
		this.width = (int)r.getWidth();
	}
	
	public int getWidth() {
		return this.width;
	}
	
	public int getHeight() {
		return this.height;
	}

	@Override
	public void mouseDragged(MouseEvent e) {}

	@Override
	public void mouseMoved(MouseEvent e) {
		this.mouseX = e.getX();
		this.mouseY = e.getY();
	}

	@Override
	public void mouseClicked(MouseEvent e) {}

	@Override
	public void mousePressed(MouseEvent e) {}

	@Override
	public void mouseReleased(MouseEvent e) {}

	@Override
	public void mouseEntered(MouseEvent e) {}

	@Override
	public void mouseExited(MouseEvent e) {}
	
}
