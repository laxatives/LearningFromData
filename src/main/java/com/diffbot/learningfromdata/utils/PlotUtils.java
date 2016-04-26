package com.diffbot.learningfromdata.utils;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;

import javax.swing.JFrame;
import javax.swing.JPanel;

import de.erichseifert.gral.graphics.Drawable;
import de.erichseifert.gral.ui.InteractivePanel;

public class PlotUtils {
	public static final Color BLUE = new Color(53, 148, 216);
	public static final Color GOLD = new Color(239, 184, 43);
	public static final Color GREEN = new Color(106, 166, 53);
	public static final Color RED = new Color(240, 127, 80);
	// Gral ignores alpha channel...
	public static final Color TRANSPARENT = new Color(0, 0, 0, 0);
	
	public static void drawPlot(Drawable plot) {
		JPanel panel = new JPanel(new BorderLayout());
		panel.setPreferredSize(new Dimension(800, 600));
		panel.setBackground(Color.WHITE);
		panel.add(new InteractivePanel(plot), BorderLayout.CENTER);
		
		JFrame frame = new JFrame();
		frame.getContentPane().add(panel, BorderLayout.CENTER);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(new Dimension(800, 600));
		frame.setVisible(true);
	}
}
