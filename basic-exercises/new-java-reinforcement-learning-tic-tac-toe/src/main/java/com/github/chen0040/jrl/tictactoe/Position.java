package com.github.chen0040.jrl.tictactoe;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Position {
    private int x;
    private int y;
    public int BOUND;

    // Explicit accessors keep compilation stable even when Lombok processing is disabled.
    public int getX() {
        return x;
    }

    public void setX(int x) {
        this.x = x;
    }

    public int getY() {
        return y;
    }

    public void setY(int y) {
        this.y = y;
    }

    public Position() {

    }

    public Position(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public int toInteger(Board board) {
        return x * board.size() + y;
    }

    public static Position fromInteger(Board board, int intValue) {
        int x = (int)Math.floor((double)intValue / board.size());
        int y = intValue - x * board.size();
        return new Position(x, y);
    }
}
