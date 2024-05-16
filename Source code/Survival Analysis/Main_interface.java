package bsm;


import bsm.core.*;

import java.awt.*;
import java.awt.event.*;
import java.io.*;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import javax.swing.event.*;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.jfree.data.category.DefaultCategoryDataset;

import java.util.*;
import java.util.List;
import java.text.*;
import java.net.*;

/**
 * Class implementing the main input interface
 */
public class Main_interface extends JFrame implements ActionListener {
	
	static final String SZDELIM = "|;,";
	static final boolean BDEBUG = false;
	
	BSM_DataSet expressiondata;
	HashMap<Integer, HashMap<String, Set<Integer>>> network;
	String[][] result;
	HashMap<String, List<Integer>> tftg;
	HashMap<Integer, List<String>> tgtf;
	// Main
	static String szoptionalFile = " Optional";
	static String szDataFileDEF = "";
	static int nstaticsourceDEF = 0;
	static int nummissing = 0;

	// GUI
	static long s1;
	static boolean bendsearch = false;
	static String szsurvival;
	static String szdataval;
	static String szmoduleval;
	

	int npathwaysourcecb;
	int ntfsourcecb;
	int nppisourcecb;
	String szuserFileField1;
	String szuserFileField2;
	String szuserFileField3;
	
	static JFileChooser theChooser = new JFileChooser();

	// Regulator Scoring GUI
	JButton regScoreFileButton = new JButton("Browse...", Util.createImageIcon("Open16.gif"));
	JButton regScoreHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JTextField regScoreField;
	static int NUMCOLS = 42;
	// Strings for the labels
	static Color gray = new Color(235,235,235);
	static Color defaultColor;

	JTextField goField;
	JTextField extraField;
	JTextField xrefField;
	JTextField categoryIDField;
	JTextField taxonField;
	JTextField evidenceField;
	JButton infoButton = new JButton(Util.createImageIcon("About16.gif"));
	static JFileChooser fc1 = new JFileChooser(new File("Data/Files for Survival Analysis/Expression data"));
	static JFileChooser fc2 = new JFileChooser(new File("Data/Files for Survival Analysis/Network file"));
	static JFileChooser fc3 = new JFileChooser(new File("Data/Files for Survival Analysis/Survival file"));
	static Container contentPane;
	JPanel textpanel;
	JPanel panel4;


	JButton survivalbutton= new JButton();
	JButton databutton= new JButton();
	JButton modulebutton= new JButton();
	JTextArea runningtext;
	JButton Run = new JButton();
	JTextField survivalField, dataField, moduleField;
	static int missinter = 0; // Set the missing value as
	static int filterduplicates = 0; // Set the duplicate genes as
	static String division;
	static double cutoff;
	
	JRadioButton log1, log2;
	static int nnormalizeDEF = 1;
	String labels1[] = {" Min value", " Mean value", " Zero"};
	JComboBox comboBox1 = new JComboBox(labels1);
	JComboBox comboBox1_case = new JComboBox(labels1);
	String labels3[] = {" Max value", " Mean value"};
	JComboBox comboBox3 = new JComboBox(labels3);
	JComboBox comboBox3_case = new JComboBox(labels3);
	ButtonGroup normGroup = new ButtonGroup();
	final JSpinner jpro, jpro2;
	

	/**
	 * Class constructor - builds the input interface calls parseDefaults to get
	 * the initial settings from a default settings file if specified
	 */
	public Main_interface() throws FileNotFoundException, IOException {
		super("MulNet_survival v.1.0");

		contentPane = getContentPane();
		BoxLayout layout = new BoxLayout(contentPane, BoxLayout.Y_AXIS);
		contentPane.setLayout(layout);

		JPanel panel1=new JPanel();
		panel1.setBorder(new TitledBorder(null,"Load Data",TitledBorder.LEFT,TitledBorder.TOP));
		//panel1.setLayout(new BorderLayout());
		panel1.setLayout(null);
     /*****************************************************************/
		
		//JLabel j1=new JLabel("Survival file source:");
		//j1.setBounds(15,20,200,35);
		//survivalsourcecb.setBounds(170, 25, 150, 25);
		
		JLabel j7=new JLabel("Load survival file:");
		j7.setBounds(15,20,300,35);
		survivalField=new JTextField(szDataFileDEF, JLabel.TRAILING);
		survivalField.setBounds(150,20,400,25);
		survivalField.setBorder(BorderFactory.createLineBorder(Color.black));
		survivalField.setOpaque(true);
		survivalField.setBackground(Color.white);
		survivalbutton.setText("Load File");
		survivalbutton.setHideActionText(true);
		survivalbutton.addActionListener(this);
		survivalbutton.setBounds(555,20,90,25);
		panel1.add(j7);
		panel1.add(survivalField);
		panel1.add(survivalbutton);
		/*****************************************************************/
		
		JLabel j8=new JLabel("Load expression data:");
		j8.setBounds(15,55,300,35);
		dataField=new JTextField(szDataFileDEF, JLabel.TRAILING);
		dataField.setBounds(150,58,400,25);
		dataField.setBorder(BorderFactory.createLineBorder(Color.black));
		dataField.setOpaque(true);
		dataField.setBackground(Color.white);
		databutton.setText("Load File");
		databutton.setHideActionText(true);
		databutton.addActionListener(this);
		databutton.setBounds(555,58,90,25);
		panel1.add(j8);
		panel1.add(dataField);
		panel1.add(databutton);
		/*****************************************************************/
		
		JLabel j9=new JLabel("Load network file:");
		j9.setBounds(15,90,300,35);
		moduleField=new JTextField(szDataFileDEF, JLabel.TRAILING);
		moduleField.setBounds(150,93,400,25);
		moduleField.setBorder(BorderFactory.createLineBorder(Color.black));
		moduleField.setOpaque(true);
		moduleField.setBackground(Color.white);
		modulebutton.setText("Load File");
		modulebutton.setHideActionText(true);
		modulebutton.addActionListener(this);
		modulebutton.setBounds(555,93,90,25);
		panel1.add(j9);
		panel1.add(moduleField);
		panel1.add(modulebutton);
		/*****************************************************************/
		JLabel ll1=new JLabel("Compare the top and bottom");
		ll1.setBounds(15,128,300,35);
		SpinnerNumberModel firstgroup = new SpinnerNumberModel(0.25, 0, 0.5, 0.01);
		jpro = new JSpinner(firstgroup);
		jpro.setBounds(185,135,45,22);
		JLabel ll2=new JLabel("patients.");
		ll2.setBounds(235,128,300,35);
		
		JLabel ll3=new JLabel("P-value cutoff for log-rank test:");
		ll3.setBounds(350,128,300,35);
		SpinnerNumberModel secondgroup = new SpinnerNumberModel(0.05, 0, 1, 0.01);
		jpro2 = new JSpinner(secondgroup);
		jpro2.setBounds(535,135,45,22);
		
		
		panel1.add(ll1);
		panel1.add(jpro);
		panel1.add(ll2);
		panel1.add(ll3);
		panel1.add(jpro2);
		/*****************************************************************/
		
        JScrollPane   scrollpanel1   =   new   JScrollPane(panel1, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED, JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        scrollpanel1.setBounds(100, 100, 745, 90);
		panel1.setPreferredSize(new Dimension(scrollpanel1.getWidth() - 50, scrollpanel1.getHeight()*2));
		
		
		textpanel=new JPanel();
		textpanel.setPreferredSize(new Dimension(745, 150));
		textpanel.setBorder(new TitledBorder(null,"Analysis",TitledBorder.LEFT,TitledBorder.TOP));
		BoxLayout layout2 = new BoxLayout(textpanel, BoxLayout.X_AXIS);
		textpanel.setLayout(layout2);
		
		panel4=new JPanel();
		panel4.setLayout(new BorderLayout());
		runningtext=new JTextArea();
		runningtext.setLineWrap(true);
		JScrollPane sp1=new JScrollPane(runningtext);
		panel4.add(sp1);
		textpanel.add(panel4);

		Run.setText("Run");
		Run.setHideActionText(true);
		Run.addActionListener(this);
		infoButton.addActionListener(this);
		infoButton.setBackground(gray);
		
		JPanel panel6=new JPanel();
		panel6.setPreferredSize(new Dimension(745, 50));
		panel6.add(Run);
		panel6.add(infoButton);
		contentPane.add(scrollpanel1);
		contentPane.add(textpanel);
		contentPane.add(panel6);
	}


	private BSM_DataSet buildset(String szexp1val, Integer maxcase, Integer missinter, boolean btakelog) throws Exception {
		BSM_DataSet theDataSet1 = new BSM_DataSet(szexp1val, 0, 0, btakelog);
		theDataSet1 = new BSM_DataSet(theDataSet1.filterMissing());	
		theDataSet1 = new BSM_DataSet(theDataSet1.maxAndFilterDuplicates());
		return theDataSet1;	
	}
	
	/**
	 * A control method that handles the response for when the execute button on
	 * the interface is pressed including building the data set, running the
	 * TSMiner modeling procedure, and displaying the results
	 */
	public void clusterscript(String szsurvival, String szdataval,
			String szmoduleval) throws Exception {
		    
		//szsurvival = "D:\\Personal issues\\MulNet\\tool\\MulNet v.1.0\\Data\\Files for Survival Analysis/Survival file/HNSC survival file.txt";
	    //szdataval = "D:\\Personal issues\\MulNet\\tool\\MulNet v.1.0\\Data\\Files for Survival Analysis\\Expression data/HNSC expression.txt";
	    //szmoduleval = "D:\\Personal issues\\MulNet\\result\\2 network\\lgtftg network/primary fibroblast lgtftg.txt";
		
				if (szsurvival.trim().equals("") ) {
					throw new IllegalArgumentException("No survival file given!");
				} else if (!(new File(szsurvival)).exists()) {
					throw new IllegalArgumentException("The survival file '" + szsurvival+ "' cannot be found.");
				} else if (szdataval.trim().equals("")) {
					throw new IllegalArgumentException("No expression data given!");
				} else if (!(new File(szdataval)).exists()) {
					throw new IllegalArgumentException("The expression data file '" + szdataval+ "' cannot be found.");
				} else if (szmoduleval.trim().equals("")) {
					throw new IllegalArgumentException("No network file given!");
				} else if (!(new File(szmoduleval)).exists()) {
					throw new IllegalArgumentException("The network file '" + szdataval+ "' cannot be found.");
				}
				
				expressiondata = buildset(szdataval, 0, 0, false);
				expressiondata.normalization_zcore();

				readNetworkFile(szmoduleval);
				Double[][] survivalfile = readsurvivalfile(szsurvival);
				int groupcount = (int) (expressiondata.numcols * Double.parseDouble(division));
				Survivalanalysis ss = new Survivalanalysis();
				HashMap<Integer, HashMap<String, Double>> pvalues = ss.survival(expressiondata, survivalfile, network, groupcount, runningtext);
				expressiondata.controldata = null;
				expressiondata.controlnorm = null;
				survivalfile = null;
				HashMap<String, HashMap<String, Set<String>>> survivalnetwork = convert(network, pvalues);
				
				try {
		        	/*
					try{
						 BufferedWriter outXml1 = new BufferedWriter(new FileWriter("D:/network survival.txt"));
						 outXml1.write("Gene"+"\t"+"Pvalue"+"\t"+"Functional unit"+"\n");
						 for(int m=0;m<result.length;m++) {
							 outXml1.write(result[m][0]+"\t");
							 outXml1.write(result[m][1]+"\t");
							 outXml1.write(result[m][2]+"\n");
						}
						outXml1.flush(); 
						outXml1.close();
						System.out.println("DONE");	
					    }catch (Exception e) {
							System.out.println("FALSE"); 
						e.printStackTrace(); 
					}
		        	*/
		        	javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {	
							JFrame frame = new JFrame("Survival Analysis Results");
							Container theDialogContainer = frame.getContentPane();
							theDialogContainer.setBackground(Color.white);
							JTabbedPane tabbedPane = new JTabbedPane();
							TSMinerGui_PathwayTable newContentPane1 = new TSMinerGui_PathwayTable(
									network, pvalues, tftg, survivalnetwork, expressiondata, result, frame);
								newContentPane1.setOpaque(true); // content panes must be opaque
								tabbedPane.addTab("Survival Table", null, newContentPane1,"Survival Table");
							
							theDialogContainer.add(tabbedPane);
							//frame.setPreferredSize(new Dimension(800, 800));
							//frame.setMinimumSize(new Dimension(800, 800));
							frame.setContentPane(theDialogContainer);
							frame.pack();
							frame.setVisible(true);
						}
					});
					
				} catch (Exception e) {
					e.printStackTrace();
			    }
		//System.exit(0);
		long e1 = System.currentTimeMillis();
		System.out.println("Time: " + (e1 - s1) + "ms");
	}
	
	public List<File> getAllFile(File dirFile) {
        if (Objects.isNull(dirFile) || !dirFile.exists() || dirFile.isFile())
            return null;

        File[] childrenFiles = dirFile.listFiles();
        if (Objects.isNull(childrenFiles) || childrenFiles.length == 0)
            return null;

        List<File> files = new ArrayList<>();
        for (File childFile : childrenFiles) {
            if (childFile.isFile()) {
                files.add(childFile);
            }
        }
        return files;
    }
	
	public void readNetworkFile(String filename){
		network = new HashMap<>();
		tftg = new HashMap<String, List<Integer>>();
		tgtf = new HashMap<Integer, List<String>>();
		try {
			BufferedReader bufferedReader1 = new BufferedReader(new FileReader(filename));   
			String lineTxt = null;
			while((lineTxt = bufferedReader1.readLine()) != null){
            	if(lineTxt.split("\t").length == 3){
            		String gene1 = lineTxt.split("\t")[0];
            		String gene2 = lineTxt.split("\t")[1];
            		String type = lineTxt.split("\t")[2];
            		if((type.equalsIgnoreCase("regulator-target") || type.equalsIgnoreCase("ligand-tf") || type.equalsIgnoreCase("tf-tg")) && expressiondata.gene2int.get(gene2) != null) {
            			int g2 = expressiondata.gene2int.get(gene2);
            			if(tftg.get(gene1) != null) {
            				List<Integer> tgs = tftg.get(gene1);
            				if(!tgs.contains(g2)) {
            					tgs.add(g2);
                				tftg.put(gene1, tgs);
            				}
            			} else {
            				List<Integer> tgs = new ArrayList<Integer>();
            				tgs.add(g2);
            				tftg.put(gene1, tgs);
            			}
            			if(tgtf.get(g2) != null) {
            				List<String> tfs = tgtf.get(g2);
            				if(!tfs.contains(gene1)) {
            					tfs.add(gene1);
            					tgtf.put(g2, tfs);
            				}
            			} else {
        					List<String> tfs = new ArrayList<String>();
        					tfs.add(gene1);
        					tgtf.put(g2, tfs);
        				}
            		} else if(expressiondata.gene2int.get(gene1) != null && expressiondata.gene2int.get(gene2) != null) {
            			int g1 = expressiondata.gene2int.get(gene1);
            			int g2 = expressiondata.gene2int.get(gene2);
            			if(network.get(g1) != null) {
            				HashMap<String, Set<Integer>> neighbors = network.get(g1);
            				if(neighbors.get(type) != null) {
            					Set<Integer> tgs = neighbors.get(type);
            					tgs.add(g2);
            					neighbors.put(type, tgs);
            					network.put(g1, neighbors);
            				} else {
            					Set<Integer> tgs = new HashSet<Integer>();
            					tgs.add(g2);
            					neighbors.put(type, tgs);
            					network.put(g1, neighbors);
            				}
            			} else {
            				HashMap<String, Set<Integer>> neighbors = new HashMap<String, Set<Integer>>();
            				Set<Integer> tgs = new HashSet<Integer>();
        					tgs.add(g2);
        					neighbors.put(type, tgs);
        					network.put(g1, neighbors);
            			}
            			
            			if(network.get(g2) != null) {
            				HashMap<String, Set<Integer>> neighbors = network.get(g2);
            				if(neighbors.get(type) != null) {
            					Set<Integer> tgs = neighbors.get(type);
            					tgs.add(g1);
            					neighbors.put(type, tgs);
            					network.put(g2, neighbors);
            				} else {
            					Set<Integer> tgs = new HashSet<Integer>();
            					tgs.add(g1);
            					neighbors.put(type, tgs);
            					network.put(g2, neighbors);
            				}
            			} else {
            				HashMap<String, Set<Integer>> neighbors = new HashMap<String, Set<Integer>>();
            				Set<Integer> tgs = new HashSet<Integer>();
        					tgs.add(g1);
        					neighbors.put(type, tgs);
        					network.put(g2, neighbors);
            			}
            			
            		}          		
            	} else {
            		System.out.println(lineTxt);
            	}
            }
            bufferedReader1.close();
    } catch (Exception e) {
    	throw new IllegalArgumentException("Read network file error.");
    }
		Collection<Integer> cl = tgtf.keySet();
		Iterator<Integer> itr = cl.iterator();
		while (itr.hasNext()) {
			int key = itr.next();
			List<String> tfs = tgtf.get(key);
			if(network.get(key) != null) {
				HashMap<String, Set<Integer>> neighbors = network.get(key);
				for(String tf:tfs) {
					List<Integer> tgs = tftg.get(tf);
					if(tgs.size() > 1) {
						Set<Integer> neighbor = new HashSet<Integer>();
						for(int tg:tgs) {
							if(tg != key) neighbor.add(tg);
						}
						neighbors.put(tf, neighbor);
						network.put(key, neighbors);
					}
				}
			} else {
				HashMap<String, Set<Integer>> neighbors = new HashMap<String, Set<Integer>>();
				for(String tf:tfs) {
					List<Integer> tgs = tftg.get(tf);
					if(tgs.size() > 1) {
						Set<Integer> neighbor = new HashSet<Integer>();
						for(int tg:tgs) {
							if(tg != key) neighbor.add(tg);
						}
						neighbors.put(tf, neighbor);
						network.put(key, neighbors);
					}
				}
			}
		}
	}
	
	public HashMap<String, HashMap<String, Set<String>>> convert (HashMap<Integer, HashMap<String, Set<Integer>>> network,
			HashMap<Integer, HashMap<String, Double>> pvalues){
		HashMap<String, HashMap<String, Set<String>>> survivalnetwork = new HashMap<String, HashMap<String, Set<String>>>();
		List<String[]> resultlist = new ArrayList<String[]>();
		
		Collection<Integer> cl = network.keySet();
		Iterator<Integer> itr = cl.iterator();
		while (itr.hasNext()) {
			int gene1 = itr.next();
			String name1 = expressiondata.genenames[gene1];
			HashMap<String, Set<Integer>> neighbor = network.get(gene1);
			HashMap<String, Double> pvalueset = pvalues.get(gene1);
			HashMap<String, Set<String>> survivalneighbor = new HashMap<String, Set<String>>();
			Collection<String> c2 = neighbor.keySet();
			Iterator<String> itr2 = c2.iterator();
			while (itr2.hasNext()) {
				String type = itr2.next();
				if(pvalueset.get(type) <= cutoff) {
					Set<Integer> targets = neighbor.get(type);
					Set<String> survivaltarget = new HashSet<String>();
					String[] row = new String[4];
					row[0] = name1;
					row[1] = pvalueset.get(type)+"";
					row[2] = targets.size()+"";
					if(tftg.keySet().contains(type)) {
						row[3] = "co-regulated by "+type;
					} else {
						row[3] = "connected through "+type;
					}
					resultlist.add(row);		
					for(int gene2:targets) {
						HashMap<String, Double> pvalueset2 = pvalues.get(gene2);
						if(pvalueset2.get(type) <= cutoff) {
							String name2 = expressiondata.genenames[gene2];
							survivaltarget.add(name2);
						}
					}
					survivalneighbor.put(type, survivaltarget);
				}
			}
			if(survivalneighbor.size()>0) {
				survivalnetwork.put(name1, survivalneighbor);
			}
		}
		result = new String[resultlist.size()][];
		for(int i=0;i<resultlist.size();i++) {
			result[i] = resultlist.get(i);
		}
		return survivalnetwork;
	}
	
	public Double[][] readsurvivalfile (String filename){
		Double[][] result = new Double[expressiondata.numcols][2];
		List<String> sample = new ArrayList<String>();
		List<String> statue = new ArrayList<String>();
		List<String> time = new ArrayList<String>();
		try {
			BufferedReader bufferedReader1 = new BufferedReader(new FileReader(filename));   
			String lineTxt = bufferedReader1.readLine();
			while((lineTxt = bufferedReader1.readLine()) != null){
				if(Arrays.asList(lineTxt.split("\t")).size() > 2){
					sample.add(Arrays.asList(lineTxt.split("\t")).get(0));
					statue.add(Arrays.asList(lineTxt.split("\t")).get(1));
					time.add(Arrays.asList(lineTxt.split("\t")).get(2));
				}
			}
			bufferedReader1.close();
		}catch (Exception e) {
	    	throw new IllegalArgumentException("Read survival file error.");
	    }
		int matched = 0;
		for(int i=0;i<expressiondata.numcols;i++) {
			String submitter = expressiondata.dsamplemins[i];
			for(int j=0;j<sample.size();j++) {
				if(sample.get(j).equals(submitter)) {
					matched++;
					if(statue.get(j).equalsIgnoreCase("dead") || statue.get(j).equalsIgnoreCase("1")) {
						result[i][0] = 1.0;
						result[i][1] = Double.parseDouble(time.get(j));
					} else if(statue.get(j).equalsIgnoreCase("alive") || statue.get(j).equalsIgnoreCase("0")) {
						result[i][0] = 0.0;
						result[i][1] = Double.parseDouble(time.get(j));
					}
					break;
				}
			}
		}
		System.out.println(matched);
		if(matched == 0) {
			throw new IllegalArgumentException("ID mathching error between the expression data and survival file.");
		}
		return result;
	}

	/**
	 * define the button methods
	 */
	public void actionPerformed(ActionEvent e) {
		Object esource = e.getSource();

		if (esource == survivalbutton) {
			int returnVal = fc3.showOpenDialog(this);
			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc3.getSelectedFile();
				survivalField.setText(file.getAbsolutePath());
			}
		}  else if (esource == databutton) {
			int returnVal = fc1.showOpenDialog(this);
			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc1.getSelectedFile();
				dataField.setText(file.getAbsolutePath());
			}
		} else if (esource == modulebutton) {
			int returnVal = fc2.showOpenDialog(this);
			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc2.getSelectedFile();
				moduleField.setText(file.getAbsolutePath());
			}
		}  else if (esource == Run) {
			s1 = System.currentTimeMillis();
			szsurvival = survivalField.getText();
			szdataval = dataField.getText(); 
			szmoduleval = moduleField.getText(); 
			division = jpro.getValue().toString();
			cutoff = (double) jpro2.getValue();
			
			int temp2=comboBox1.getSelectedIndex();
			if(temp2==0){
				missinter=0;
			}else if(temp2==1){
				missinter=1;
			}else{
				missinter=2;
			}
			
			int temp3=comboBox3.getSelectedIndex();
			if(temp3==0){
				filterduplicates=0;
			}else if(temp3==1){
				filterduplicates=1;
			}
			this.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
			
			final JFrame fframe = this;
			Runnable clusterrun = new Runnable() {
				public void run() {
					Run.setEnabled(false);
					try {
						clusterscript(szsurvival, szdataval, szmoduleval);
					} catch (IllegalArgumentException iex) {
						final IllegalArgumentException fiex = iex;
						iex.printStackTrace(System.out);

						javax.swing.SwingUtilities.invokeLater(new Runnable() {
							public void run() {
								JOptionPane.showMessageDialog(fframe, fiex.getMessage(), "Error",JOptionPane.ERROR_MESSAGE);
							}
						});
					} catch (Exception ex) {
						final Exception fex = ex;

						javax.swing.SwingUtilities.invokeLater(new Runnable() {
							public void run() {
								JOptionPane.showMessageDialog(fframe, fex.toString(), "Exception thrown",JOptionPane.ERROR_MESSAGE);
								fex.printStackTrace(System.out);
							}
						});
					}
					Run.setEnabled(true); 
					fframe.setCursor(Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
				}
			};
			(new Thread(clusterrun)).start();
			
			
		} else if (esource == infoButton) {//The Information and Help button
			String szMessage = "This is version 1.0.0 of network-based survival analysis.\n"
					+ "It is available under a GPL v3.0 license.\n"
					+ "Any questions or bugs found should "
					+ "be emailed to free1234hm@163.com.";

			Util.renderDialog(this, szMessage, 50, 100, "Information");
		}
	}
	
	/**
	 * Create the GUI and show it. For thread safety, this method should be
	 * invoked from the event-dispatching thread.
	 */
	private static void createAndShowGUI() throws FileNotFoundException,IOException {
		// Make sure we have nice window decorations.
		// JFrame.setDefaultLookAndFeelDecorated(true);
		// Create and set up the window.
		JFrame frame = new Main_interface();
		frame.setLocation(10, 25);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		// Display the window.
		frame.pack();
		frame.setVisible(true);
	}

	
	/**
	 * The main method which when executed will have the input interface created
	 */
	public static void main(String[] args) throws Exception {
			javax.swing.SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					try {
						createAndShowGUI();
					} catch (FileNotFoundException ex) {
						ex.printStackTrace(System.out);
					} catch (IOException ex) {
						ex.printStackTrace(System.out);
					}
				}
			});
	}
}
