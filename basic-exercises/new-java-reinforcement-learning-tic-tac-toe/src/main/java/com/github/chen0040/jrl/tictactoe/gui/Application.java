package com.github.chen0040.jrl.tictactoe.gui;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.EventQueue;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;

import com.github.chen0040.jrl.tictactoe.Board;

/**
 * Swing Game initialization.
 * Application
 */
public class Application extends JFrame {

    public Application() {

        initUI();
    }

    private void initUI() {

        final Game game = new Game(new Board());
        this.add(game, BorderLayout.CENTER);

        final JPanel commands = new JPanel(new BorderLayout());
        add(commands, BorderLayout.SOUTH);

        final JButton btnStart = new JButton("Q-Train");
        btnStart.addActionListener(e -> {
            game.trainQ(learner -> {
                JOptionPane.showMessageDialog(Application.this,
                        "Training completed");
            });
        });
        commands.add(btnStart, BorderLayout.WEST);
        final JButton btnAccelerateLearning = new JButton("SARSA Train");
        btnAccelerateLearning.addActionListener(e -> {
            game.trainSarsa(learner -> {
                JOptionPane.showMessageDialog(Application.this,
                        "Training completed");
            });
        });
        commands.add(btnAccelerateLearning, BorderLayout.CENTER);

        final JButton btnStop = new JButton("New Game");
        btnStop.addActionListener(e -> game.newGame());
        commands.add(btnStop, BorderLayout.EAST);

        this.setTitle("Application");
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setLocationRelativeTo(null);
        this.setSize(new Dimension(Game.SCREEN_WIDTH + 20, Game.SCREEN_HEIGHT + 80));

    }

    public static void main(String[] args) {

        EventQueue.invokeLater(() -> {
            Application ex = new Application();
            ex.setVisible(true);
        });
    }
}
