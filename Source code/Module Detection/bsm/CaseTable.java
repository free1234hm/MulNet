package bsm;

import heatmapframe.GetHeatMap;
import heatmapframe.HeatMap;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.SpinnerNumberModel;

import java.awt.Color;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.*;
import java.util.List;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.awt.datatransfer.*;

import javax.swing.border.TitledBorder;
import javax.swing.table.*;

import bsm.core.TableModelST;
import bsm.core.TableSorter;
import bsm.core.Util;

/**
 * Class for a table that shows enrichment of TF targets along a path
 */
public class CaseTable extends JPanel implements ActionListener {
	JFrame theframe;
	BSM_DataSet theDataSet;
	LearningCase casemodules;
	private JFrame saveImageFrame;
	JPanel mapPanel, mapPanel1, mapPanel2;
	JPanel tablePanel;
	String[] columnNames;
	String[] tableheader;
	String[] tableheader2;
	String[][] tabledata;
	JButton saveSet;
	JButton saveHeatMap;
	JButton showNetwork;
	JButton saveNetwork;
	JButton saveModel;
	JButton pathwayButton;
	JButton pathwayButton2;
	JButton showButton;
	String showText;
	boolean showheatmap;;
	String[][] positable;
	String[][] negatable;
	String[][] totaltable;
	HeatMap posiheatmap;
	HeatMap negaheatmap;
	int selectchild;
	
	JLabel j2 = new JLabel("Min count: ");
	SpinnerNumberModel count = new SpinnerNumberModel(new Integer(10), new Integer(0), null, new Integer(1));
	JSpinner num1 = new JSpinner(count);
	//num1.setMinimumSize(new Dimension(20,20));
	JLabel j3 = new JLabel("Min percentage:");
	SpinnerNumberModel per = new SpinnerNumberModel(new Integer(10), new Integer(0), null, new Integer(1));
	JSpinner num2 = new JSpinner(per);
	//num2.setMinimumSize(new Dimension(20,20));
	JLabel j4 = new JLabel("%");
	JScrollPane scrollPane;
	JScrollPane scrollposi;
	JScrollPane scrollnega;
	TableSorter sorter;
	TableSorter sorterposi;
	TableSorter sorternega;
	JTable table;
	JTable tableposi;
	JTable tablenega;
	final static Color bgColor = Color.white;
	final static Color fgColor = Color.black;
	int numrows;
	LearningCase.Casenode ptr;
	int colnum;
	NumberFormat nf;
	NumberFormat nf2;
	boolean bsplit;
	DecimalFormat df1;
	DecimalFormat df2;
	
	/**
	 * Constructor - builds the table
	 */
	public CaseTable(JFrame frame1, BSM_DataSet theds, LearningCase casemodules, LearningCase.Casenode ptr) {
		// assuming that if called with root node then only one child so stats
		this.theframe = frame1;
		this.theDataSet = theds;
		this.casemodules = casemodules;
		this.ptr = ptr;
		df1 = new DecimalFormat("#0.00");
		df2 = new DecimalFormat("#0.0000");
		
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		setBackground(bgColor);
		setForeground(fgColor);

		nf2 = NumberFormat.getInstance(Locale.ENGLISH);
		nf2.setMinimumFractionDigits(2);
		nf2.setMaximumFractionDigits(2);
		
		numrows = casemodules.resultList.length;
		colnum = casemodules.resultList[0].length;
		tabledata = new String[numrows][colnum];
		
		columnNames = new String[colnum];
		columnNames[0] = "Module ID";
		columnNames[1] = "Total genes";
		columnNames[2] = "Subset1";
		columnNames[3] = "Subset2";
		for(int i=0;i<casemodules.theDataSet.dsamplemins.length;i++) {
			columnNames[i+4] = casemodules.theDataSet.dsamplemins[i];
		}
		
		tableheader = new String[casemodules.theDataSet.numcols+1];
		tableheader2 = new String[casemodules.theDataSet.numcols+2];
		tableheader[0] = "ID";
		for(int i=1;i<tableheader.length;i++) tableheader[i] = casemodules.theDataSet.dsamplemins[i-1];
		tableheader2[0] = "ID";
		tableheader2[1] = "Correlation";
		for(int i=2;i<tableheader2.length;i++) tableheader2[i] = casemodules.theDataSet.dsamplemins[i-2];
		
		if(casemodules.resultList!=null)
		{
			for (int nrow = 0; nrow < numrows; nrow++) {
				for(int ncol = 0; ncol<4; ncol++) tabledata[nrow][ncol] = casemodules.resultList[nrow][ncol];
				for(int ncol = 4; ncol<colnum; ncol++) tabledata[nrow][ncol] 
						= df1.format(Double.parseDouble(casemodules.resultList[nrow][ncol]));
			}
		}		
		
		sorter = new TableSorter(new TableModelST(tabledata, columnNames));
		table = new JTable(sorter);
		sorter.setTableHeader(table.getTableHeader());
		//table.setPreferredScrollableViewportSize(new Dimension(0, 
		//	Math.min((table.getRowHeight() + table.getRowMargin())* table.getRowCount(), 400)));

		table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
		for(int i=0;i<colnum;i++) {
			TableColumn column = table.getColumnModel().getColumn(i);
			column.setMinWidth(50);
		}
		
		scrollPane = new JScrollPane(table);
		scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
		scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
		
		tablePanel = new JPanel();
		tablePanel.setBorder(new TitledBorder(null,"Module information",TitledBorder.LEFT,TitledBorder.TOP));
		BoxLayout layout = new BoxLayout(tablePanel, BoxLayout.Y_AXIS);
		tablePanel.setLayout(layout);
		tablePanel.add(scrollPane);
		
		addheatmap();
		addButton1();
		add(tablePanel);
		addBottom();
		
		this.addComponentListener(new ComponentAdapter(){  //dynamic column size;
			public void componentResized(ComponentEvent e){
				for(int i=0;i<colnum;i++) {
					TableColumn column = table.getColumnModel().getColumn(i);
					column.setPreferredWidth(Math.max(50, theframe.getWidth()/colnum));
				}
		  }
	   });
		
		table.addMouseListener(new MouseAdapter() {
		     public void mouseClicked(MouseEvent e) {
		    	 if(table.getSelectedRow() != -1){
		    		 mapPanel1.removeAll();
		    		 mapPanel1.repaint();
		    		 mapPanel2.removeAll();
		    		 mapPanel2.repaint();
		    		 selectchild = Integer.parseInt((String) table.getValueAt(table.getSelectedRow(), 0))-1;
		    		 List<Integer> posigenes = ptr.nextptr[selectchild].posi_genelist;
		    		 List<Integer> negagenes = ptr.nextptr[selectchild].nega_genelist;
		    		 GetHeatMap heatmap = new GetHeatMap();
		    		 
		    		 if(posigenes != null && posigenes.size() > 0) {
		    			 positable = new String[posigenes.size()][casemodules.theDataSet.numcols+1];
		    			 double[][] normposi = new double[posigenes.size()][casemodules.theDataSet.numcols];	 
		    			 for(int i=0;i<posigenes.size();i++){
		    				 int geneindex = posigenes.get(i);
		    				 double largest = Double.MIN_VALUE;
						     double smallest = Double.MAX_VALUE;
		    				 positable[i][0] = casemodules.theDataSet.genenames[geneindex];
		    				 for(int j=1;j<positable[0].length;j++){
		    					 positable[i][j] = df1.format(casemodules.theDataSet.controldata[geneindex][j-1])+"";
		    					 largest = Math.max(casemodules.theDataSet.controldata[geneindex][j-1], largest);
								 smallest = Math.min(casemodules.theDataSet.controldata[geneindex][j-1], smallest);
		    				 }
		    				 double range = largest - smallest;
		    				 for(int j=0;j<casemodules.theDataSet.numcols;j++){
		    					 normposi[i][j] = (casemodules.theDataSet.controldata[geneindex][j]-smallest)/range;
		    				 }
				    	}
		    			 try {
		    				 posiheatmap = heatmap.heatmap(normposi);
		    				 } catch (Exception e2) {
		    					 e2.printStackTrace();
		    				}
		    			    sorterposi = new TableSorter(new TableModelST(positable, tableheader));
			    			tableposi = new JTable(sorterposi);
			    			sorterposi.setTableHeader(tableposi.getTableHeader());
			    			tableposi.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
			    			TableColumn column;
			    			for(int i=0;i<tableheader.length;i++){
			    				column = tableposi.getColumnModel().getColumn(i);
			    				column.setMaxWidth(120);
			    			}
			    			scrollposi = new JScrollPane(tableposi);
			    			scrollposi.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
			       			scrollposi.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
			       			
			       			if(showheatmap){
				    			saveSet.setEnabled(false);
								saveHeatMap.setEnabled(true);
								mapPanel1.add(posiheatmap);
				    		 }else{
				    			saveSet.setEnabled(true);
							    saveHeatMap.setEnabled(false);
				    			mapPanel1.add(scrollposi);
				    		 }
		    		 }
		    		 if(negagenes != null && negagenes.size() > 0) {
		    			 negatable = new String[negagenes.size()][casemodules.theDataSet.numcols+1];
		    			 double[][] normnega = new double[negagenes.size()][casemodules.theDataSet.numcols];	 
		    			 for(int i=0;i<negagenes.size();i++){
		    				 int geneindex = negagenes.get(i);
		    				 double largest = Double.MIN_VALUE;
						     double smallest = Double.MAX_VALUE;
						     negatable[i][0] = casemodules.theDataSet.genenames[geneindex];
		    				 for(int j=1;j<negatable[0].length;j++){
		    					 negatable[i][j] = df1.format(casemodules.theDataSet.controldata[geneindex][j-1])+"";
		    					 largest = Math.max(casemodules.theDataSet.controldata[geneindex][j-1], largest);
								 smallest = Math.min(casemodules.theDataSet.controldata[geneindex][j-1], smallest);
		    				 }
		    				 double range = largest - smallest;
		    				 for(int j=0;j<casemodules.theDataSet.numcols;j++){
		    					 normnega[i][j] = (casemodules.theDataSet.controldata[geneindex][j]-smallest)/range;
		    				 }
				    	}
		    			  
		    			 try {
								negaheatmap = heatmap.heatmap(normnega);
							} catch (Exception e2) {
								e2.printStackTrace();
						}
		    			 
		    			 sorternega = new TableSorter(new TableModelST(negatable, tableheader));
			    			tablenega = new JTable(sorternega);
			    			sorternega.setTableHeader(tablenega.getTableHeader());
			    			tablenega.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
			    			TableColumn column;
			    			for(int i=0;i<tableheader.length;i++){
			    				column = tablenega.getColumnModel().getColumn(i);
			    				column.setMaxWidth(120);
			    			}
			    			scrollnega = new JScrollPane(tablenega);
			    			scrollnega.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
			    			scrollnega.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
			    			
			    			if(showheatmap){
				    			saveSet.setEnabled(false);
								saveHeatMap.setEnabled(true);
								mapPanel2.add(negaheatmap);
				    		 }else{
				    			saveSet.setEnabled(true);
							    saveHeatMap.setEnabled(false);
				    			mapPanel2.add(scrollnega);
				    		 }
		    		 }
		    		 mapPanel1.updateUI();
		    		 mapPanel2.updateUI();
		    	 }
		     }
		});
	}

  private void addheatmap() {
		mapPanel = new JPanel();
		mapPanel.setBorder(new TitledBorder(null,"Expression Matrix",TitledBorder.LEFT,TitledBorder.TOP));
		BoxLayout layout = new BoxLayout(mapPanel, BoxLayout.X_AXIS);
		mapPanel.setLayout(layout);
		mapPanel.setPreferredSize(new Dimension(0, 400));
		add(mapPanel);
		
		mapPanel1 = new JPanel();
		  mapPanel1.setBorder(new TitledBorder(null,"Subset 1",TitledBorder.CENTER,TitledBorder.TOP));
		  BoxLayout layout1 = new BoxLayout(mapPanel1, BoxLayout.X_AXIS);
		  mapPanel1.setLayout(layout1);
		  mapPanel1.setPreferredSize(new Dimension(0, 400));
		  mapPanel.add(mapPanel1);
		  
		  mapPanel2 = new JPanel();
		  mapPanel2.setBorder(new TitledBorder(null,"Subset 2",TitledBorder.CENTER,TitledBorder.TOP));
		  BoxLayout layout2 = new BoxLayout(mapPanel2, BoxLayout.X_AXIS);
		  mapPanel2.setLayout(layout2);
		  mapPanel2.setPreferredSize(new Dimension(0, 400));
		  mapPanel.add(mapPanel2);
	}
	
	/**
	 * Helper function that adds information displayed at the bottom of the
	 * table information window
	 */
  
  private void addButton1() {
	  
	    showheatmap = false;
		if (showheatmap) {
			showText = "HeatMap";
		} else {
			showText = "GeneList";
		}
		showButton = new JButton(showText);
		showButton.setActionCommand("show");
		showButton.setMinimumSize(new Dimension(800, 20));
		showButton.addActionListener(this);
	  
  	    saveSet = new JButton("Save Gene Table", Util.createImageIcon("Save16.gif"));
    	saveSet.setActionCommand("savegeneset");
  	    saveSet.setMinimumSize(new Dimension(800, 20));
  	    saveSet.addActionListener(this);
		
		saveHeatMap = new JButton("Save Heatmap", Util.createImageIcon("Save16.gif"));
		saveHeatMap.setActionCommand("saveheatmap");
		saveHeatMap.setMinimumSize(new Dimension(800, 20));
		saveHeatMap.addActionListener(this);
		
		JPanel buttonPanel = new JPanel();
		buttonPanel.add(showButton);
		buttonPanel.add(saveSet);
		buttonPanel.add(saveHeatMap);
		saveSet.setEnabled(false);
		saveHeatMap.setEnabled(false);
		buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(buttonPanel);
	}
  
	private void addBottom() {

		showNetwork = new JButton("Network Map");
		showNetwork.setActionCommand("shownetwork");
		showNetwork.setMinimumSize(new Dimension(800, 20));
		showNetwork.addActionListener(this);

		saveModel = new JButton("Save Module", Util.createImageIcon("Save16.gif"));
		saveModel.setActionCommand("savemodel");
		saveModel.setMinimumSize(new Dimension(800, 20));
		saveModel.addActionListener(this);
		
		saveNetwork = new JButton("Save Network", Util.createImageIcon("Save16.gif"));
		saveNetwork.setActionCommand("savenetwork");
		saveNetwork.setMinimumSize(new Dimension(800, 20));
		saveNetwork.addActionListener(this);
		
		pathwayButton = new JButton("Enrichment Analysis");
		pathwayButton.setActionCommand("enrichment");
		pathwayButton.setMinimumSize(new Dimension(800, 20));
		pathwayButton.addActionListener(this);
		
		pathwayButton2 = new JButton("Min. Enrichment Pvalues");
		pathwayButton2.setActionCommand("minpvalue");
		pathwayButton2.setMinimumSize(new Dimension(800, 20));
		pathwayButton2.addActionListener(this);
		
		
		
		JPanel buttonPanel = new JPanel();
		buttonPanel.add(pathwayButton);
		buttonPanel.add(pathwayButton2);
		buttonPanel.add(showNetwork);
		buttonPanel.add(saveModel);
		buttonPanel.add(saveNetwork);
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

	public void printModel(PrintWriter pw) {
		pw.print("Gene"+"\t"+"Module"+"\t"+"Correlation"+"\t");
		for(int i=0;i<theDataSet.numcols-1;i++) {
			pw.print(theDataSet.dsamplemins[i]+"\t");
		}
		pw.print(theDataSet.dsamplemins[theDataSet.numcols-1]+"\n");
		
		for(int i=0;i<ptr.numchildren;i++){
			List<Integer> posigenelist = ptr.nextptr[i].posi_genelist;
			List<Integer> negagenelist = ptr.nextptr[i].nega_genelist;
			if(posigenelist.size()>0){
				for(int j=0;j<posigenelist.size();j++){
					int geneindex = posigenelist.get(j);
					pw.print(theDataSet.genenames[geneindex]+"\t"+"Module_"+(i+1)+"\t"+"1"+"\t");
			    	for(int m=0;m<theDataSet.numcols-1;m++) {
			    		pw.print(theDataSet.controldata[geneindex][m]+"\t");
			    	}
			    	pw.print(theDataSet.controldata[geneindex][theDataSet.numcols-1]+"\n");
			    }
			}
			if(negagenelist.size()>0){
				for(int j=0;j<negagenelist.size();j++){
					int geneindex = negagenelist.get(j);
					pw.print(theDataSet.genenames[geneindex]+"\t"+"Module_"+(i+1)+"\t"+"-1"+"\t");
			    	for(int m=0;m<theDataSet.numcols-1;m++) {
			    		pw.print(theDataSet.controldata[geneindex][m]+"\t");
			    	}
			    	pw.print(theDataSet.controldata[geneindex][theDataSet.numcols-1]+"\n");
			    }
			}
		}
    }
	
	public void printNetwork(PrintWriter pw) {
		for(int i=0; i<ptr.numchildren; i++) {
			 Set<String> directnetwork =ptr.nextptr[i].directnetwork;
			 Set<String> coexpressionnetwork = ptr.nextptr[i].coexpressionnetwork;
			 HashMap<String, Set<String>> tftg = ptr.nextptr[i].tftg;
			 pw.print("ID1"+"\t"+"ID2"+"\t"+"Type"+"\n");
			 for(String tt:directnetwork) {
				 pw.print(tt+"\n");
			 }
		Collection<String> cl = tftg.keySet();
		Iterator<String> itr = cl.iterator();
		while (itr.hasNext()) {
			String key = itr.next();
			Set<String> index = tftg.get(key);
			for(String gene2:index) {
				pw.print(key+"\t"+gene2+"\t"+"regulator-target"+"\n");
				}
			}
		for(String tt:coexpressionnetwork) {
			pw.print(tt+"\n");
		}
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

	public void printSet(PrintWriter pw) {
		
		if(table.getSelectedRow() != -1){
			selectchild = Integer.parseInt((String) table.getValueAt(table.getSelectedRow(), 0))-1;
			List<Integer> posigenelist = ptr.nextptr[selectchild].posi_genelist;
			List<Integer> negagenelist = ptr.nextptr[selectchild].nega_genelist;
			totaltable = new String[posigenelist.size()+negagenelist.size()][casemodules.theDataSet.numcols+2];
			int index = 0;
			if(posigenelist.size() > 0) {
				for(int i=0;i<posigenelist.size();i++){
					totaltable[index][0] = casemodules.theDataSet.genenames[posigenelist.get(i)];
					totaltable[index][1] = "1";
					for(int j=0;j<casemodules.theDataSet.numcols;j++){
						totaltable[index][j+2] = casemodules.theDataSet.controldata[posigenelist.get(i)][j]+"";
					}
					index++;
				}
			}
			if(negagenelist.size() > 0) {
				for(int i=0;i<negagenelist.size();i++){
					totaltable[index][0] = casemodules.theDataSet.genenames[negagenelist.get(i)];
					totaltable[index][1] = "-1";
					for(int j=0;j<casemodules.theDataSet.numcols;j++){
						totaltable[index][j+2] = casemodules.theDataSet.controldata[negagenelist.get(i)][j]+"";
					}
					index++;
				}
			}
			for (int ncol = 0; ncol < tableheader2.length-1; ncol++) {
				pw.print(tableheader2[ncol] + "\t");
			}
			pw.println(tableheader2[tableheader2.length - 1]);
			for (int nrow = 0; nrow < totaltable.length; nrow++){
				for (int ncol = 0; ncol < totaltable[nrow].length - 1; ncol++) {
					pw.print(totaltable[nrow][ncol]+"\t");
				}
				pw.println(totaltable[nrow][totaltable[nrow].length-1]);
			}
		}
	}
	
	/**
	 * Responds to buttons being pressed on the interface
	 */
	public void actionPerformed(ActionEvent e) {
		
		String szCommand = e.getActionCommand();
		if (szCommand.equals("copy")) {
			writeToClipboard();
		} else if (szCommand.equals("savegeneset")){
			try {
				int nreturnVal = Main_interface.theChooser.showSaveDialog(this);
				if (nreturnVal == JFileChooser.APPROVE_OPTION) {
					File f = Main_interface.theChooser.getSelectedFile();
					PrintWriter pw = new PrintWriter(new FileOutputStream(f));
					if (szCommand.equals("savegeneset")) {
						printSet(pw);
					}
					pw.close();
				}
			}catch (final FileNotFoundException fex) {
				javax.swing.SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						JOptionPane.showMessageDialog(null, fex.getMessage(),
								"Exception thrown", JOptionPane.ERROR_MESSAGE);
					}
				});
				fex.printStackTrace(System.out);
			}
		} else if(szCommand.equals("saveheatmap")){
			javax.swing.SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					saveImageFrame = new JFrame("Save as Image");
					saveImageFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
					saveImageFrame.setLocation(400,300);
					TSMinerGui_SaveImage newContentPane = new TSMinerGui_SaveImage(saveImageFrame, posiheatmap, negaheatmap);
					newContentPane.setOpaque(true);
					saveImageFrame.setContentPane(newContentPane);
					saveImageFrame.pack();
					saveImageFrame.setVisible(true);
				}
			});
		} else if (szCommand.equals("show")) {
			if(table.getSelectedRow() != -1){
				if (showheatmap) {
					showheatmap = false;
					if (showheatmap) {
						showText = "HeatMap";
					} else {
						showText = "GeneList";
					}
					showButton.setText(showText);
					if(showheatmap){
						saveSet.setEnabled(false);
						saveHeatMap.setEnabled(true);
					}else{
						saveSet.setEnabled(true);
						saveHeatMap.setEnabled(false);
					}
				} else {
					showheatmap = true;
					if (showheatmap) {
						showText = "HeatMap";
					} else {
						showText = "GeneList";
					}
					showButton.setText(showText);
					if(showheatmap){
						saveSet.setEnabled(false);
						saveHeatMap.setEnabled(true);
					}else{
						saveSet.setEnabled(true);
						saveHeatMap.setEnabled(false);
					}
				}
				Showheatmap();
			} else {
				JOptionPane.showMessageDialog(theframe, 
						"Please select a row in 'Module information'."
						, "Information", JOptionPane.INFORMATION_MESSAGE); 
			}
		} else if (szCommand.equals("shownetwork")) {
			int selectrow = table.getSelectedRow();
			if(selectrow >= 0) {
				selectchild = Integer.parseInt((String) table.getValueAt(selectrow, 0))-1;
				Set<String> directnetwork = ptr.nextptr[selectchild].directnetwork;
				HashMap<String, Set<String>> tftg = ptr.nextptr[selectchild].tftg;
				javax.swing.SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						drawNet dn = new drawNet();
						dn.drawPic(directnetwork, tftg, theDataSet);
					}
				});
			} else {
				JOptionPane.showMessageDialog(theframe, 
						"Please select a row in 'Module information'."
						, "Information", JOptionPane.INFORMATION_MESSAGE); 
			}
		} else if (szCommand.equals("savemodel")) {
			try {
				int nreturnVal = Main_interface.theChooser.showSaveDialog(this);
				if (nreturnVal == JFileChooser.APPROVE_OPTION) {
					File f = Main_interface.theChooser.getSelectedFile();
					PrintWriter pw = new PrintWriter(new FileOutputStream(f));
					if (szCommand.equals("savemodel")) {
						printModel(pw);
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
		} else if (szCommand.equals("savenetwork")) {
			try {
				int nreturnVal = Main_interface.theChooser.showSaveDialog(this);
				if (nreturnVal == JFileChooser.APPROVE_OPTION) {
					File f = Main_interface.theChooser.getSelectedFile();
					PrintWriter pw = new PrintWriter(new FileOutputStream(f));
					if (szCommand.equals("savenetwork")) {
						printNetwork(pw);
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
		} else if (szCommand.equals("enrichment")) {
			int selectrow = table.getSelectedRow();
			if(selectrow >= 0) {
				selectchild = Integer.parseInt((String) table.getValueAt(selectrow, 0))-1;
				//System.out.println(selectrow+" "+selectchild);
				if(casemodules.pathwaydata != null){
					List<String[][]> tablelist = ptr.nextptr[selectchild].enrichment;
					javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							JFrame pathwayframe = new JFrame("Enrichment analysis");
							JDialog frame = new JDialog(pathwayframe, "Enrichment analysis", true);
							frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
							frame.setLocation(20, 50);
							Container theDialogContainer = frame.getContentPane();
							theDialogContainer.setBackground(Color.white);
							JTabbedPane tabbedPane = new JTabbedPane();
							boolean miss = true;
							for(int i=0;i<tablelist.size();i++) {
								String[][] table = tablelist.get(i);
								if(table!=null &&table.length>0){
									miss = false;
									TSMinerGui_PathwayTable newContentPane1 = new TSMinerGui_PathwayTable(
											table, frame, pathwayframe);
									newContentPane1.setOpaque(true); // content panes must be opaque
									tabbedPane.addTab("Table "+i, null, newContentPane1,"TF");
									theDialogContainer.add(tabbedPane);
								}
							}
							if(miss){
								JOptionPane.showMessageDialog(null, 
										"No gene annotations identified, check the gene annotation file or "
										+ "gene IDs please.", "Information", JOptionPane.INFORMATION_MESSAGE); 
							}
							pathwayframe.setContentPane(theDialogContainer);
							pathwayframe.pack();
							pathwayframe.setVisible(true);
						}
					});
				}else{
					JOptionPane.showMessageDialog(theframe, 
							"No enrichment analysis file identified."
							, "Information", JOptionPane.INFORMATION_MESSAGE); 
				}
			} else {
				JOptionPane.showMessageDialog(theframe, 
						"Please select a row in 'Module information'."
						, "Information", JOptionPane.INFORMATION_MESSAGE); 
			}
		} else if (szCommand.equals("minpvalue")) {
				if(casemodules.minenrichmenttable != null){
					List<String[][]> tablelist = casemodules.minenrichmenttable;
					javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							JFrame pathwayframe = new JFrame("Min enrichment pvalues");
							JDialog frame = new JDialog(pathwayframe, "Min enrichment pvalues", true);
							frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
							frame.setLocation(20, 50);
							Container theDialogContainer = frame.getContentPane();
							theDialogContainer.setBackground(Color.white);
							JTabbedPane tabbedPane = new JTabbedPane();
							boolean miss = true;
							for(int i=0;i<tablelist.size();i++) {
								String[][] table = tablelist.get(i);
								if(table!=null &&table.length>0){
									miss = false;
									TSMinerGui_PathwayTable newContentPane1 = new TSMinerGui_PathwayTable(
											table, frame, pathwayframe);
									newContentPane1.setOpaque(true); // content panes must be opaque
									tabbedPane.addTab("Table "+i, null, newContentPane1,"TF");
									theDialogContainer.add(tabbedPane);
								}
							}
							if(miss){
								JOptionPane.showMessageDialog(null, 
										"No gene annotations identified, check the gene annotation file or "
										+ "gene IDs please.", "Information", JOptionPane.INFORMATION_MESSAGE); 
							}
							pathwayframe.setContentPane(theDialogContainer);
							pathwayframe.pack();
							pathwayframe.setVisible(true);
						}
					});
				}else{
					JOptionPane.showMessageDialog(theframe, 
							"No gene annotation file input."
							, "Information", JOptionPane.INFORMATION_MESSAGE); 
				}
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


    
    public String[][] BubbleSort_increase(String[][] r, Integer n, Integer col) //����ð������
 	{
 		 int low = 0;   
 		    int high= n - 1; //���ñ����ĳ�ʼֵ  
 		    String[] tmp;
 		    int j;  
 		    while (low < high) {  
 		        for (j= low; j< high; ++j) //����ð��,�ҵ������  
 		            if (Double.parseDouble(r[j][col])> Double.parseDouble(r[j+1][col])) {  
 		                tmp = r[j]; r[j]=r[j+1];r[j+1]=tmp;  
 		            }   
 		        --high;                 //�޸�highֵ, ǰ��һλ  
 		        for ( j=high; j>low; --j) //����ð��,�ҵ���С��  
 		            if (Double.parseDouble(r[j][col])<Double.parseDouble(r[j-1][col])) {  
 		                tmp = r[j]; r[j]=r[j-1];r[j-1]=tmp;  
 		            }  
 		        ++low; //�޸�lowֵ,����һλ  
 		    }   
 	return r;
 	}
    
    public String[][] BubbleSort_decrease(String[][] r, Integer n, Integer col) //����ð������
 	{
 		 int low = 0;   
 		    int high= n - 1; //���ñ����ĳ�ʼֵ  
 		    String[] tmp;
 		    int j;  
 		    while (low < high) {  
 		        for (j= low; j< high; ++j) //����ð��,�ҵ������  
 		            if (Double.parseDouble(r[j][col])< Double.parseDouble(r[j+1][col])) {  
 		                tmp = r[j]; r[j]=r[j+1];r[j+1]=tmp;  
 		            }   
 		        --high;                 //�޸�highֵ, ǰ��һλ  
 		        for ( j=high; j>low; --j) //����ð��,�ҵ���С��  
 		            if (Double.parseDouble(r[j][col])>Double.parseDouble(r[j-1][col])) {  
 		                tmp = r[j]; r[j]=r[j-1];r[j-1]=tmp;  
 		            }  
 		        ++low; //�޸�lowֵ,����һλ  
 		    }   
 	return r;
 	}

    public Integer getrank(List<Double> pvalue, double value){
    	int cc = 1;
    	for(int i=0;i<pvalue.size();i++){
    		if(pvalue.get(i)<value){
    			cc++;
    		}
    	}
    	return cc;
    }
    private void Showheatmap() {
        mapPanel1.removeAll();
       	mapPanel1.repaint();
       	mapPanel2.removeAll();
      	mapPanel2.repaint();
       	if(showheatmap){
       	  if(posiheatmap != null) mapPanel1.add(posiheatmap);
       	  if(negaheatmap != null) mapPanel2.add(negaheatmap);
       	}else{
       	  if(scrollposi != null) mapPanel1.add(scrollposi);
       	  if(scrollnega != null) mapPanel2.add(scrollnega);
       	}
        mapPanel1.updateUI();
		mapPanel2.updateUI();
	}
    
}