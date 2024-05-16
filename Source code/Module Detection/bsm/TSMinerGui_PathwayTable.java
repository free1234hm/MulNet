package bsm;

import bsm.core.TableModelST;
import bsm.core.TableSorter;
import bsm.core.Util;


import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.util.*;
import java.util.List;
import java.text.NumberFormat;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.awt.datatransfer.*;

import javax.swing.*;
import javax.swing.table.*;


/**
 * Class for a table that shows enrichment of TF targets along a path
 */
public class TSMinerGui_PathwayTable extends JPanel implements ActionListener {
	String[][] ResultList;
	int ndepth;
	HashMap<String, List<String>> pathwaygene;
	HashMap<String, List<List<String>>> tf_tg;
	HashMap<String, List<List<String>>> tf_pg;
	JDialog dialog;
	JFrame theframe;
	String[] columnNames;
	String[][] tabledata;
	JButton copyButton;
	JButton saveButton;
	JButton mapButton;
	JScrollPane scrollPane;
	TableSorter sorter;
	JTable table;
	final static Color bgColor = Color.white;
	final static Color fgColor = Color.black;
	int numrows;
	int colnum;
	NumberFormat nf;
	NumberFormat nf2;
	NumberFormat df1;
	NumberFormat df2;
	
	/**
	 * Constructor - builds the table
	 */
	public TSMinerGui_PathwayTable(String[][] ResultList, JDialog dialog, JFrame frame1) {

		this.dialog = dialog;
		this.theframe = frame1;
		this.ResultList = ResultList;
		
		df1 = NumberFormat.getInstance(Locale.ENGLISH);
		df1.setMinimumFractionDigits(2);
		df1.setMaximumFractionDigits(2);
		df2 = NumberFormat.getInstance(Locale.ENGLISH);
		df2.setMinimumFractionDigits(4);
		df2.setMaximumFractionDigits(4);
		
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		setBackground(bgColor);
		setForeground(fgColor);

		numrows = ResultList.length-1;
		colnum = ResultList[0].length;
		nf2 = NumberFormat.getInstance(Locale.ENGLISH);
		nf2.setMinimumFractionDigits(2);
		nf2.setMaximumFractionDigits(2);
	
			columnNames = ResultList[0];
			tabledata = new String[numrows][colnum];
			
			for (int nrow = 0; nrow < numrows; nrow++) {
				for(int ncol = 0; ncol<colnum; ncol++){
					tabledata[nrow][ncol] = ResultList[nrow+1][ncol];
				}
			}

			sorter = new TableSorter(new TableModelST(tabledata, columnNames));
			table = new JTable(sorter);
			sorter.setTableHeader(table.getTableHeader());
			
			table.setPreferredScrollableViewportSize(new Dimension(800, Math.min((table.getRowHeight() + table.getRowMargin())
							* table.getRowCount(), 400)));

			table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
			TableColumn column;
			for(int i=0;i<colnum;i++) {
				column = table.getColumnModel().getColumn(i);
				column.setPreferredWidth(100);
			}
			column = table.getColumnModel().getColumn(colnum-1);
			column.setMaxWidth(300);
			
			/*
			for (int ncolindex = 5; ncolindex < columnNames.length-1; ncolindex++) {
				column = table.getColumnModel().getColumn(ncolindex);
				column.setPreferredWidth(120);
			}
			column = table.getColumnModel().getColumn(columnNames.length-1);
			column.setPreferredWidth(120);
			*/
		

		scrollPane = new JScrollPane(table);
		add(scrollPane);

		addBottom();
		
		this.addComponentListener(new ComponentAdapter(){  //dynamic column size;
			public void componentResized(ComponentEvent e){
				for(int i=0;i<colnum;i++) {
					TableColumn column = table.getColumnModel().getColumn(i);
					column.setPreferredWidth(Math.max(50, theframe.getWidth()/colnum-5));
				}
		  }
	   });
	}

	/**
	 * Helper function that adds information displayed at the bottom of the
	 * table information window
	 */
	private void addBottom() {

		nf = NumberFormat.getInstance(Locale.ENGLISH);
		nf.setMinimumFractionDigits(3);
		nf.setMaximumFractionDigits(3);


		copyButton = new JButton("Copy Table", Util.createImageIcon("Copy16.gif"));
		copyButton.setActionCommand("copy");
		copyButton.setMinimumSize(new Dimension(800, 20));
		copyButton.addActionListener(this);

		saveButton = new JButton("Save Table", Util.createImageIcon("Save16.gif"));
		saveButton.setActionCommand("save");
		saveButton.setMinimumSize(new Dimension(800, 20));
		saveButton.addActionListener(this);

		JPanel buttonPanel = new JPanel();
		buttonPanel.setBackground(Color.white);

		buttonPanel.add(copyButton);
		buttonPanel.add(saveButton);

		buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(buttonPanel);	
	}

	/**
	 * Writes the content of the table to a file specified through pw
	 */
	public void printFile(PrintWriter pw) {
		for (int ncol = 0; ncol < columnNames.length - 1; ncol++) {
			pw.print(columnNames[ncol] + "\t");
		}
		pw.println(columnNames[columnNames.length - 1]);

		for (int nrow = 0; nrow < tabledata.length; nrow++) {
			for (int ncol = 0; ncol < tabledata[nrow].length - 1; ncol++) {
				pw.print(sorter.getValueAt(nrow, ncol) + "\t");
			}
			pw.println(sorter.getValueAt(nrow, columnNames.length - 1));
		}
	}

	/**
	 * Copies the content of the table to the clipboard
	 */
	public void writeToClipboard() {
		StringBuffer sbuf = new StringBuffer();
		for (int ncol = 0; ncol < columnNames.length - 1; ncol++) {
			sbuf.append(columnNames[ncol] + "\t");
		}
		sbuf.append(columnNames[columnNames.length - 1] + "\n");

		for (int nrow = 0; nrow < tabledata.length; nrow++) {
			for (int ncol = 0; ncol < tabledata[nrow].length - 1; ncol++) {
				sbuf.append(sorter.getValueAt(nrow, ncol) + "\t");
			}
			sbuf.append(sorter.getValueAt(nrow, columnNames.length - 1) + "\n");
		}
		// get the system clipboard
		Clipboard systemClipboard = Toolkit.getDefaultToolkit()
				.getSystemClipboard();
		// set the textual content on the clipboard to our
		// Transferable object

		Transferable transferableText = new StringSelection(sbuf.toString());
		systemClipboard.setContents(transferableText, null);
	}

	/**
	 * Responds to buttons being pressed on the interface
	**/
	public void actionPerformed(ActionEvent e) {
		String szCommand = e.getActionCommand();
		if (szCommand.equals("copy")) {
			writeToClipboard();
		} else if (szCommand.equals("save")) {
			try {
				int nreturnVal = Main_interface.theChooser.showSaveDialog(this);
				if (nreturnVal == JFileChooser.APPROVE_OPTION) {
					File f = Main_interface.theChooser.getSelectedFile();
					PrintWriter pw = new PrintWriter(new FileOutputStream(f));
					if (szCommand.equals("save")) {
						printFile(pw);
					}
					pw.close();
				}
			} catch (final FileNotFoundException fex) {
				javax.swing.SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						JOptionPane.showMessageDialog(null, fex.getMessage(),
								"Exception thrown", JOptionPane.ERROR_MESSAGE);
					}
				});
				fex.printStackTrace(System.out);
			}
		} else if (szCommand.equals("help")) {
			String szMessage = "This table gives the annotation information about gene modules."
					+ "  Consult section 4.13 of the user manual for more details on this table.  ";
			Util.renderDialog(dialog, szMessage);// textArea);
		}
	}

	/**
	 * Converts the value of dval to a String that is displayed on the table
	 */
	public static String doubleToSz(double dval) {
		String szexp;
		double dtempval = dval;
		int nexp = 0;

		NumberFormat nf2 = NumberFormat.getInstance(Locale.ENGLISH);
		nf2.setMinimumFractionDigits(3);
		nf2.setMaximumFractionDigits(3);

		NumberFormat nf1 = NumberFormat.getInstance(Locale.ENGLISH);
		nf1.setMinimumFractionDigits(2);
		nf1.setMaximumFractionDigits(2);

		if (dval <= 0) {
			szexp = "0.000";
		} else {
			while ((dtempval < 0.9995) && (dtempval > 0)) {
				nexp--;
				dtempval = dtempval * 10;
			}

			if (nexp < -2) {
				dtempval = Math.pow(10, Math.log(dval) / Math.log(10) - nexp);
				szexp = nf1.format(dtempval) + "e" + nexp;

			} else {
				szexp = nf2.format(dval);
			}
		}

		return szexp;
	}
	
	class Pathway_pval {
		String pathwayID;
		double pvalue;
		Pathway_pval(String pathwayID, double pvalue) {
			this.pathwayID = pathwayID;
			this.pvalue = pvalue;
		}
	}
    
}