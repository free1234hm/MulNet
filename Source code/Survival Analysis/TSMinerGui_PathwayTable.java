package bsm;

import bsm.core.TableModelST;
import bsm.core.TableSorter;
import bsm.core.Util;

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
	int ndepth;
	HashMap<Integer, HashMap<String, Set<Integer>>> network;
	HashMap<Integer, HashMap<String, Double>> pvalues;
	HashMap<String, HashMap<String, Set<String>>> survivalnetwork;
	HashMap<String, List<Integer>> tftg;
	BSM_DataSet expressiondata;
	JFrame theframe;
	String[] columnNames;
	String[][] tabledata;
	
	JButton copyButton;
	JButton saveButton1;
	JButton saveButton2;
	JButton drawButton;
	JScrollPane scrollPane;
	TableSorter sorter;
	JTable table;
	final static Color bgColor = Color.white;
	final static Color fgColor = Color.black;
	int numrows;
	int colnum;
	
	/**
	 * Constructor - builds the table
	 */
	public TSMinerGui_PathwayTable(HashMap<Integer, HashMap<String, Set<Integer>>> network,
			HashMap<Integer, HashMap<String, Double>> pvalues, HashMap<String, List<Integer>> tftg,
			HashMap<String, HashMap<String, Set<String>>> survivalnetwork, BSM_DataSet expressiondata, 
			String[][] ResultList, JFrame frame1) {
		
		this.theframe = frame1;
		this.network = network;
		this.pvalues = pvalues;
		this.survivalnetwork = survivalnetwork;
		this.tabledata = ResultList;
		this.tftg = tftg;
		this.expressiondata = expressiondata;
		
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		setBackground(bgColor);
		setForeground(fgColor);

		numrows = ResultList.length;
		colnum = ResultList[0].length;
	
		columnNames = new String[4];
		columnNames[0] = "Gene";
		columnNames[1] = "Pvalue";
		columnNames[2] = "Partner";
		columnNames[3] = "Type";
			
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


		copyButton = new JButton("Copy Table", Util.createImageIcon("Copy16.gif"));
		copyButton.setActionCommand("copy");
		copyButton.setMinimumSize(new Dimension(800, 20));
		copyButton.addActionListener(this);

		saveButton1 = new JButton("Save Table", Util.createImageIcon("Save16.gif"));
		saveButton1.setActionCommand("save1");
		saveButton1.setMinimumSize(new Dimension(800, 20));
		saveButton1.addActionListener(this);
		
		saveButton2 = new JButton("Save Network", Util.createImageIcon("Save16.gif"));
		saveButton2.setActionCommand("save2");
		saveButton2.setMinimumSize(new Dimension(800, 20));
		saveButton2.addActionListener(this);
		
		drawButton = new JButton("Show Network", Util.createImageIcon("Save16.gif"));
		drawButton.setActionCommand("show");
		drawButton.setMinimumSize(new Dimension(800, 20));
		drawButton.addActionListener(this);

		JPanel buttonPanel = new JPanel();
		buttonPanel.setBackground(Color.white);

		buttonPanel.add(copyButton);
		buttonPanel.add(saveButton1);
		buttonPanel.add(saveButton2);
		buttonPanel.add(drawButton);

		buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(buttonPanel);	
	}

	/**
	 * Writes the content of the table to a file specified through pw
	 */
	public void printTable(PrintWriter pw) {
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
		} else if (szCommand.equals("save1")) {
			try {
				int nreturnVal = Main_interface.theChooser.showSaveDialog(this);
				if (nreturnVal == JFileChooser.APPROVE_OPTION) {
					File f = Main_interface.theChooser.getSelectedFile();
					PrintWriter pw = new PrintWriter(new FileOutputStream(f));
					if (szCommand.equals("save1")) {
						printTable(pw);
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
		} else if (szCommand.equals("save2")) {
			Set<String> result = new HashSet<String>();
			Collection<String> cl = survivalnetwork.keySet();
			Iterator<String> itr = cl.iterator();
			while (itr.hasNext()) {
				String gene1 = itr.next();
				HashMap<String, Set<String>> neighbor = survivalnetwork.get(gene1);
				Collection<String> c2 = neighbor.keySet();
				Iterator<String> itr2 = c2.iterator();
				while (itr2.hasNext()) {
					String type = itr2.next();
					if(tftg.keySet().contains(type)) {
						result.add(type+"\t"+gene1+"\t"+"regulator-target");
					} else {
						Set<String> targets = neighbor.get(type);
						if(targets!=null && targets.size()>0) {
							for(String gene2:targets) {
								String row = getrow(gene1, gene2, type);
								result.add(row);
							}
						}
					}
				}
			}
			try {
				int nreturnVal = Main_interface.theChooser.showSaveDialog(this);
				if (nreturnVal == JFileChooser.APPROVE_OPTION) {
					File f = Main_interface.theChooser.getSelectedFile();
					PrintWriter pw = new PrintWriter(new FileOutputStream(f));
					if (szCommand.equals("save2")) {
						pw.print("ID1" + "\t");
						pw.print("ID2" + "\t");
						pw.println("Type");
						for (String row:result) pw.println(row);
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
			
		} else if (szCommand.equals("show")) {
			int[] rows = table.getSelectedRows();
			if(rows!=null && rows.length>=0) {
				List<String> genes = new ArrayList<String>();
				List<String> types = new ArrayList<String>();
				for(int i=0;i<rows.length;i++){
					genes.add(table.getValueAt(rows[i],0).toString());
					types.add(table.getValueAt(rows[i],3).toString().replace("co-regulated by ","").replace("connected through ", ""));
				}
				javax.swing.SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						drawNet dn = new drawNet();
						dn.drawPic(genes, types, network, survivalnetwork, tftg, expressiondata);
					}
				});
			} else {
				JOptionPane.showMessageDialog(theframe, 
						"Please select a row in 'Survival Table'."
						, "Information", JOptionPane.INFORMATION_MESSAGE); 
			}
			
		}
	}
	private String getrow(String gene1, String gene2, String type) {
		if(gene1.compareTo(gene2) < 0) {
			return(gene1+"\t"+gene2+"\t"+type);
		} else {
			return(gene2+"\t"+gene1+"\t"+type);
		}
	}
	/**
	 * Converts the value of dval to a String that is displayed on the table
	 */
	public String doubleToSz(double dval) {
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
    
}