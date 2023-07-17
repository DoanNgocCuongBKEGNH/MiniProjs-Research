package game.controls.player;

import java.util.*;
import java.util.concurrent.*;
import java.lang.Math;

import game.controls.board.*;

public class Player {
	private ArrayList<Stone> inHand;
	private ArrayList<Stone> taken;
	private int playerId;
	private int dir = -1;
	private int curIndex = -1;
	private int penalty = 0;
	
	public Player(int id) {
		this.inHand = new ArrayList<Stone>();
		this.taken = new ArrayList<Stone>();
		this.playerId = id;
	}
	
	public void reset() {
		this.inHand = new ArrayList<Stone>();
		this.taken = new ArrayList<Stone>();
		curIndex = -1;
		penalty = 0;
	}
	
	public void makeMove(Board b) {
		if(isTurn()) {
			int nextIndex = Math.floorMod(this.curIndex+dir, 12);
			int afterIndex = Math.floorMod(this.curIndex+2*dir, 12);
			BoardCell cur = b.getCells()[curIndex];
			BoardCell next = b.getCells()[nextIndex];
			BoardCell after = b.getCells()[afterIndex];
			int mc = moveCase(b, cur, next, after);
			switch(mc) {
			case 0:
				releaseStone(cur);
				break;
			case 1:
				pickupStones(cur);
				break;
			case 2:
				takeStonesInNext(next, mc==2);
				break;
			case 3:
				takeStonesInNext(next, mc==2);
				break;
			case 4:
				this.curIndex = -1;
				break;
			}
		}
		try {
			TimeUnit.MILLISECONDS.sleep(300);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public boolean isTurn() {
		return (this.curIndex>=0);
	}
	
	public void raiSoi(Board b) {
		if(playerId == 1) {
			for(int i=0; i<5; i++) {
				b.getCells()[i].getStonesInCell().add(new Stone(false));
			}
		}
		if(playerId == 2) {
			for(int i=6; i<11; i++) {
				b.getCells()[i].getStonesInCell().add(new Stone(false));
			}
		}
		this.penalty ++;
	}
	
	public void pickupStones(BoardCell bc) {
		ArrayList<Stone> cur = bc.getStonesInCell();
		if(bc.isOQuan()) {
			this.curIndex = -1;
		}
		else {
			this.inHand.addAll(cur);
			cur.clear();
			this.curIndex = Math.floorMod((curIndex+dir), 12);
		}
	}
	
	public void releaseStone(BoardCell bc) {
		//System.out.println(this.inHand.size());
		if(this.inHand.size()>0) {
			ArrayList<Stone> cur = bc.getStonesInCell();
			cur.add(this.inHand.get(this.inHand.size()-1));
			this.inHand.remove(this.inHand.size()-1);
			//System.out.println(this.inHand.size());
			this.curIndex = Math.floorMod(this.curIndex+dir, 12);
		}
	}
	
	public void takeStonesInNext(BoardCell next, boolean endTurn) {
		ArrayList<Stone> stones = next.getStonesInCell();
		this.taken.addAll(stones);
		stones.clear();
		if(endTurn) {
			this.curIndex = -1;
		}
		else {
			this.curIndex = Math.floorMod(this.curIndex+2*dir, 12);
		}
	}
	
	public int moveCase(Board b, BoardCell cur, BoardCell next, BoardCell after) { 
		int ret = 0;
		
		if(this.inHand.size() > 0) {
			ret = 0;//Release 1 stone to cell
		}
		
		if(this.inHand.size() == 0 && cur.getNumberOfStones()>0) {
			ret = 1;//Pickup all stones in cell
		}
		
		if(this.inHand.size() == 0 && cur.getNumberOfStones() == 0 && next.getNumberOfStones() > 0 && after.getNumberOfStones()>0) {
			ret = 2;//Eat all stones in next cell and end turn
		}
		
		if(this.inHand.size() == 0 && cur.getNumberOfStones() == 0 && next.getNumberOfStones() > 0 && after.getNumberOfStones()==0) {
			ret = 3;//Eat all stones in next cell and continue
		}
		
		if(this.inHand.size() == 0 && cur.getNumberOfStones() == 0 && next.getNumberOfStones() == 0) {
			ret = 4;//End turn
		}
		
		return ret;
	}
	
	public int getPoint() {
		int ret = 0;
		for(Stone s: this.taken) {
			ret += s.getValue();
		}
		return ret - this.penalty*5;
	}
	
	public int getNumberTakenStones() {
		return this.taken.size();
	}
	
	public void setCurIndex(int curIndex) {
		this.curIndex = curIndex;
	}
	
	public void setDir(int dir) {
		this.dir = dir;
	}
	
	public boolean isValidMove(int ci) {
		boolean ret = true;
		switch(this.playerId) {
		case 1:
			if(ci<0 || ci>4) {
				ret = false;
			}
			
			break;
		case 2:
			if(ci<6 || ci>10) {
				ret = false;
			}
		}
		return ret;
	}
	
	public void moveSetup(int ci, int d) {
		if(isValidMove(ci)) {
			this.curIndex = ci;
			this.dir = d;
		}
	}
	
	public int getPlayerId() {
		return this.playerId;
	}
}
