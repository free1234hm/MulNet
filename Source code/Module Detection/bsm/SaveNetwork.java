package bsm;

import javax.swing.*;

import edu.uci.ics.jung.graph.SparseMultigraph;
import edu.uci.ics.jung.graph.util.EdgeType;

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.*;


/**
 * Class to encapsulate window used to specify a file to save a TSMiner model
 */ 
public class SaveNetwork extends JPanel
{
	/**
	 * Class constructor
	 */
	public SaveNetwork(final JFrame theFrame, SparseMultigraph<String, String> graph)
	{
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		String[] sznames = ImageIO.getWriterFormatNames();
		JFileChooser theChooser = new JFileChooser();
		add(theChooser);
		theChooser.setDialogType(JFileChooser.SAVE_DIALOG);
		theChooser.addActionListener(new ActionListener() 
		{
			public void actionPerformed(ActionEvent e) 
			{
				// set label's icon to the current image
				String state = (String)e.getActionCommand();

				if (state.equals(JFileChooser.CANCEL_SELECTION))
				{
					theFrame.setVisible(false);
					theFrame.dispose();
				}
				else if (state.equals(JFileChooser.APPROVE_SELECTION))
				{
					File file = theChooser.getSelectedFile();
					try
					{
						PrintWriter pw = new PrintWriter(new FileOutputStream(file));
						pw.print("Name"+"\t"+"Name"+"\t"+"Type"+"\n");
						for(String str : graph.getEdges()) {
							if(graph.getEdgeType(str) == EdgeType.DIRECTED) {
								pw.print(str+"\t"+"regulator-target"+"\n");
							} else if(graph.getEdgeType(str) == EdgeType.UNDIRECTED) {
								pw.print(str+"\t"+"direct"+"\n");
							}
						}
						pw.close();
					    
					}catch (final IOException fex){
						javax.swing.SwingUtilities.invokeLater(new Runnable() 
						{
							public void run() 
							{
								JOptionPane.showMessageDialog(null, fex.getMessage(), 
										"Exception thrown", JOptionPane.ERROR_MESSAGE);
							}
						});
						fex.printStackTrace(System.out);
					}
					theFrame.setVisible(false);
					theFrame.dispose();
				}
			}
		});			      
	}
}