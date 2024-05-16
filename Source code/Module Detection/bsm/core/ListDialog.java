package bsm.core;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.io.*;
import javax.swing.filechooser.*;
import java.awt.image.*;

/*
 * Classed used for the repeat dialog interface. It is modeled after this sample code:
 * http://java.sun.com/docs/books/tutorial/uiswing/examples/components/ListDialogRunnerProject/src/components/ListDialog.java
 * here is the copyright for that code
 *
 * Copyright (c) 1995 - 2008 Sun Microsystems, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *   - Neither the name of Sun Microsystems nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
public class ListDialog extends JDialog implements ActionListener {

//	private static ListDialog dialog;
//	private static String value = "";
	private JList list;
	public Vector<String> data;
	JButton mainButton;
	JButton removeButton;
	JButton okButton;
	JFileChooser fc;
	public static Color buttonColor = new Color(255, 246, 143);

	JFrame frame;
//	Vector<String> names = new Vector<String>();

	/**
	 * Constructors the Repeat interface
	 */
	public ListDialog(JFrame frame, Vector<String> data,
			JButton mainButton, 
			JFileChooser fc) {
		super(frame, "Known Module Files", false);

		this.data = data;
		this.frame = frame;
		this.mainButton = mainButton;
		this.fc = fc;
		okButton = new JButton("OK");
		okButton.setActionCommand("ok");
		okButton.addActionListener(this);

		okButton.setPreferredSize(new Dimension(145, 28));
		okButton.setMaximumSize(new Dimension(145, 28));
		removeButton = new JButton("Remove File");
		removeButton.setPreferredSize(new Dimension(145, 28));
		removeButton.setMaximumSize(new Dimension(145, 28));
		removeButton.setActionCommand("remove");
		removeButton.addActionListener(this);
		removeButton.setEnabled((data.size() > 0));

		final JButton setButton = new JButton("Add File",
				Util.createImageIcon("Open16.gif"));
		setButton.setActionCommand("add");
		setButton.addActionListener(this);
		setButton.setPreferredSize(new Dimension(145, 28));
		setButton.setMaximumSize(new Dimension(145, 28));
		getRootPane().setDefaultButton(setButton);

		// main part of the dialog
		list = new JList(data) {
			// Note from the sample
			// Subclass JList to workaround bug 4832765, which can cause the
			// scroll pane to not let the user easily scroll up to the beginning
			// of the list. An alternative would be to set the unitIncrement
			// of the JScrollBar to a fixed value. You wouldn't get the nice
			// aligned scrolling, but it should work.
			public int getScrollableUnitIncrement(Rectangle visibleRect,
					int orientation, int direction) {
				int row;
				if (orientation == SwingConstants.VERTICAL && direction < 0
						&& (row = getFirstVisibleIndex()) != -1) {
					Rectangle r = getCellBounds(row, row);
					if ((r.y == visibleRect.y) && (row != 0)) {
						Point loc = r.getLocation();
						loc.y--;
						int prevIndex = locationToIndex(loc);
						Rectangle prevR = getCellBounds(prevIndex, prevIndex);

						if (prevR == null || prevR.y >= r.y) {
							return 0;
						}
						return prevR.height;
					}
				}
				return super.getScrollableUnitIncrement(visibleRect,
						orientation, direction);
			}
		};

		list.setVisibleRowCount(-1);
		list.addMouseListener(new MouseAdapter() {
			public void mouseClicked(MouseEvent e) {
				if (e.getClickCount() == 2) {
					setButton.doClick(); // emulate button click
				}
			}
		});
		list.setSelectedIndex(0);
		JScrollPane listScroller = new JScrollPane(list);
		listScroller.setPreferredSize(new Dimension(600, 150));
		listScroller.setAlignmentX(LEFT_ALIGNMENT);

		// Create a container so that we can add a title around
		// the scroll pane. Can't add a title directly to the
		// scroll pane because its background would be white.
		// Lay out the label and scroll pane from top to bottom.
		JPanel listPane = new JPanel();
		listPane.setLayout(new BoxLayout(listPane, BoxLayout.PAGE_AXIS));
		JLabel label = new JLabel("Repeat Data File(s):");
		label.setLabelFor(list);
		listPane.add(label);

		listPane.add(listScroller);
		listPane.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
		listPane.setAlignmentX(CENTER_ALIGNMENT);
		// Lay out the buttons from left to right.
		JPanel buttonPane = new JPanel(new SpringLayout());

		buttonPane.add(setButton);
		buttonPane.add(removeButton);
		buttonPane.add(okButton);

		SpringUtilities.makeCompactGrid(buttonPane, 1, 3, 6, 6, 6, 6);

		// Put everything together, using the content pane's BorderLayout.
		Container contentPane = getContentPane();
		BoxLayout layout = new BoxLayout(contentPane, BoxLayout.PAGE_AXIS);
		contentPane.setLayout(layout);
		// Initialize values.


		contentPane.add(listPane);
		contentPane.add(buttonPane);
		pack();
	}

	/**
	 * Handle clicks on the Set and Cancel buttons.
	 */
	public void actionPerformed(ActionEvent e) {
		String szcommand = e.getActionCommand();
		if (szcommand.equals("ok")) {
			setVisible(false);
		} else if (szcommand.equals("remove")) {
			Object[] nAselectindex = list.getSelectedValues();
			for (int aindex = 0; aindex < nAselectindex.length; aindex++) {
				Object selectedval = nAselectindex[aindex];
				if (selectedval != null) {
					data.remove(selectedval);
				}
			}
			if (data.size() < 1) {
				removeButton.setEnabled(false);
			} else if (list.getSelectedIndex() >= data.size()) {
				list.setSelectedIndex(data.size() - 1);
			}
		} else if (szcommand.equals("add")) {
			fc.setMultiSelectionEnabled(true);
			int returnVal = fc.showOpenDialog(this);
			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File[] file = fc.getSelectedFiles();
				if(file.length>0) {
					for(int i=0;i<file.length;i++) {
						if(!data.contains(file[i].getAbsolutePath())) {
							data.add(file[i].getAbsolutePath());
							removeButton.setEnabled(true);
							mainButton.setBackground(ListDialog.buttonColor);
							list.setSelectedIndex(data.size() - 1);
						}
					}
				}
			}
		}
		list.updateUI();
	}
	
	/**
	 * Updates the repeat list interface based on the contents of updateData and
	 * bupdatealltime
	 */
	public void updateSettings(Vector<String> updateData, boolean bupdatealltime) {

		list.setSelectedIndex(0);
		if (updateData.size() > 0) {
			removeButton.setEnabled(true);
			mainButton.setBackground(ListDialog.buttonColor);
		} else {
			removeButton.setEnabled(false);
		}

		data.removeAllElements();
		data.addAll(updateData);
		list.updateUI();
	}
}
