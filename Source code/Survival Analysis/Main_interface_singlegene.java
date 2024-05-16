package bsm;


import bsm.core.*;
import javastat.survival.inference.LogRankTest;
import javastat.util.DataManager;

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
public class Main_interface_singlegene extends JFrame implements ActionListener {
	
	static final String SZDELIM = "|;,";
	static final boolean BDEBUG = false;
	HashMap<String, Integer> geneindex = new HashMap<String, Integer>();
	HashMap<Integer, List<Integer>> pair = new HashMap<Integer, List<Integer>>();
	HashMap<Integer, List<String>> type = new HashMap<Integer, List<String>>();
	BSM_DataSet expressiondata;
	HashMap<Integer, List<Integer>> network;
	HashMap<Integer, List<String>> edgetype;
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
	static JFileChooser fc1 = new JFileChooser(new File("Survival analysis file"));

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
	
	JRadioButton log1, log2;
	static int nnormalizeDEF = 1;
	String labels1[] = {" Min value", " Mean value", " Zero"};
	JComboBox comboBox1 = new JComboBox(labels1);
	JComboBox comboBox1_case = new JComboBox(labels1);
	String labels3[] = {" Max value", " Mean value"};
	JComboBox comboBox3 = new JComboBox(labels3);
	JComboBox comboBox3_case = new JComboBox(labels3);
	ButtonGroup normGroup = new ButtonGroup();
	final JSpinner jpro;
	

	/**
	 * Class constructor - builds the input interface calls parseDefaults to get
	 * the initial settings from a default settings file if specified
	 */
	public Main_interface_singlegene() throws FileNotFoundException, IOException {
		super("CREAM_survival v.1.0");

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
		panel1.add(ll1);
		panel1.add(jpro);
		panel1.add(ll2);
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
		    
		szsurvival = "D:\\Personal issues\\CREAM\\CREAM tool\\CREAM v.1.1\\Data/6_Survival analysis file/Survival file (test)/Survival file (COAD).txt";
	    szdataval = "D:\\Personal issues\\CREAM\\result\\4_Module_based process\\3_survival genes\\Survival files/Survival_expression_COAD.txt";
	    szmoduleval = "D:\\Personal issues\\CREAM\\result\\4_Module_based process\\2_robust network/RobustNetwork_coregulation.txt";
		
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
				for(int i=0;i<expressiondata.genenames.length;i++){
					geneindex.put(expressiondata.genenames[i].toUpperCase(), i);
				}
				Double[][] survivalfile = readsurvivalfile(szsurvival);
				
				//readFile(szmoduleval);
				int groupcount = (int) (expressiondata.numcols * Double.parseDouble(division));
				String[][] result = new String[expressiondata.numrows][2];
				for(int m=0;m<expressiondata.numrows;m++) {
					result[m][0] = expressiondata.genenames[m];
					double[] exp1 = expressiondata.controldata[m];
					String[][] expvalue = new String[expressiondata.numcols][2];
					for(int i=0;i<exp1.length;i++) {
						expvalue[i][0] = i+"";
						expvalue[i][1] = exp1[i]+"";
					}
					Util.BubbleSort_inc(expvalue, expvalue.length, 1);
					List<Double> timehigh0 = new ArrayList<Double>();
					List<Double> timelow0 = new ArrayList<Double>();
					List<Double> censorhigh0 = new ArrayList<Double>();
					List<Double> censorlow0 = new ArrayList<Double>();
					int countlow0 = 0;
					int counthigh0 = expressiondata.numcols-1;
					while(timelow0.size() < groupcount && countlow0 < expressiondata.numcols) {
						if(survivalfile[Integer.parseInt(expvalue[countlow0][0])][0] != null) {
							censorlow0.add(survivalfile[Integer.parseInt(expvalue[countlow0][0])][0]);
							timelow0.add(survivalfile[Integer.parseInt(expvalue[countlow0][0])][1]);
						}
						countlow0++;
					}
					while(timehigh0.size() < groupcount && counthigh0 >= 0) {
						if(survivalfile[Integer.parseInt(expvalue[counthigh0][0])][0] != null) {
							censorhigh0.add(survivalfile[Integer.parseInt(expvalue[counthigh0][0])][0]);
							timehigh0.add(survivalfile[Integer.parseInt(expvalue[counthigh0][0])][1]);
						}
						counthigh0--;
					}
					double[] aa0 = timehigh0.stream().mapToDouble(i->i).toArray();
					double[] bb0 = censorhigh0.stream().mapToDouble(i->i).toArray();
					double[] cc0 = timelow0.stream().mapToDouble(i->i).toArray();
					double[] dd0 = censorlow0.stream().mapToDouble(i->i).toArray();
					DataManager dm1 = new DataManager();
					LogRankTest testclass1 = new LogRankTest(aa0, bb0, cc0, dd0);
					result[m][1] = dm1.roundDigits(testclass1.pValue, 7.0)+"";
					
				}
				
				try{
					 BufferedWriter outXml1 = new BufferedWriter(new FileWriter("D:/single gene survial.txt"));
					 outXml1.write("Gene"+"\t"+"Num"+"\t"+"\n");
					 for(int m=0;m<result.length;m++) {
						 for(int n=0;n<result[m].length;n++) {
							 outXml1.write(result[m][n]+"\t");
						 }
						 outXml1.newLine();
					}
					outXml1.flush(); 
					outXml1.close();
					System.out.println("DONE");	
				    }catch (Exception e) {
						System.out.println("FALSE"); 
					e.printStackTrace(); 
				}
				//readNetworkFile(szmoduleval);
				

				try {
					
		        	
					
		        	/*
		        	javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							JFrame frame1 = new JFrame("Survival Analysis Results");
							Container theDialogContainer = frame1.getContentPane();
							theDialogContainer.setBackground(Color.white);
							JTabbedPane tabbedPane = new JTabbedPane();
								CaseTable newContentPane1 = new CaseTable(frame1, 
										finalcase, finalcase.ModuleSet, expressiondata);
								newContentPane1.setOpaque(true); // content panes must be opaque
								tabbedPane.addTab("Resulting Modules", null, newContentPane1,"TF Enrichment");
							
							theDialogContainer.add(tabbedPane);
							frame1.setPreferredSize(new Dimension(800, 800));
							frame1.setMinimumSize(new Dimension(800, 800));
							frame1.setContentPane(theDialogContainer);
							frame1.pack();
							frame1.setVisible(true);
						}
					});
					*/
				} catch (Exception e) {
					e.printStackTrace();
			}
				
				/*
				String[] datasample = expressiondata.dsamplemins;
				List<Integer> sampleindex = readfile();

				String series1 = "Highly co-expressed";  
			    String series2 = "Lowly co-expressed";  
			    DefaultCategoryDataset dataset = new DefaultCategoryDataset();  
			    dataset.addValue(200, series1, "1");  
			    dataset.addValue(150, series1, "2");  
			    dataset.addValue(100, series1, "3");  
			    dataset.addValue(210, series1, "4");  
			    dataset.addValue(240, series1, "5");
			    dataset.addValue(195, series1, "6");  
			    dataset.addValue(245, series1, "7");  
			    
			    dataset.addValue(150, series2, "1");  
			    dataset.addValue(130, series2, "2");  
			    dataset.addValue(95, series2, "3");  
			    dataset.addValue(195, series2, "4");  
			    dataset.addValue(200, series2, "5");  
			    dataset.addValue(180, series2, "6");  
			    dataset.addValue(230, series2, "7"); 
				
				try {
		        	javax.swing.SwingUtilities.invokeLater(new Runnable() {
		        		public void run() {
		        			String title = "Kaplan-Meier survival plot";
		        			String pvalue = "Log-rank test p-value = 0.03";
		        			SurvivalCurve survivaltPane1 = new SurvivalCurve(title, dataset, pvalue);
							survivaltPane1.setAlwaysOnTop(true);  
							survivaltPane1.pack();  
							survivaltPane1.setSize(600, 500);
							//survivaltPane1.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);  
							survivaltPane1.setVisible(true);  
						}
					});
		            } catch (Exception e) {
						e.printStackTrace();
				}
		*/
		//System.exit(0);
		long e1 = System.currentTimeMillis();
		System.out.println("Time: " + (e1 - s1) + "ms");
	}
	
	public List<File> getAllFile(File dirFile) {
        // 如果文件夹不存在或着不是文件夹，则返回 null
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
	public void readFile(String filename){
		try {
			BufferedReader bufferedReader1 = new BufferedReader(new FileReader(filename));   
			String lineTxt = bufferedReader1.readLine();
            while((lineTxt = bufferedReader1.readLine()) != null){
            	if(lineTxt.split("\t").length > 2){
            		if(geneindex.get(lineTxt.split("\t")[0]) != null && geneindex.get(lineTxt.split("\t")[1]) != null) {
            			int gene1 = geneindex.get(lineTxt.split("\t")[0]);
                		int gene2 = geneindex.get(lineTxt.split("\t")[1]);
                		String tt = lineTxt.split("\t")[2];
                		
                		if(pair.get(gene1) != null) {
                			List<Integer> list1 = pair.get(gene1);
                			List<String> list2 = type.get(gene1);
                			list1.add(gene2);
                			list2.add(tt);
                			pair.put(gene1, list1);
                			type.put(gene1, list2);
                		} else {
                			List<Integer> list1 = new ArrayList<Integer>();
                			List<String> list2 = new ArrayList<String>();
                			list1.add(gene2);
                			list2.add(tt);
                			pair.put(gene1, list1);
                			type.put(gene1, list2);
                		}
                		
                		if(pair.get(gene2) != null) {
                			List<Integer> list1 = pair.get(gene2);
                			List<String> list2 = type.get(gene2);
                			list1.add(gene1);
                			list2.add(tt);
                			pair.put(gene2, list1);
                			type.put(gene2, list2);
                		} else {
                			List<Integer> list1 = new ArrayList<Integer>();
                			List<String> list2 = new ArrayList<String>();
                			list1.add(gene1);
                			list2.add(tt);
                			pair.put(gene2, list1);
                			type.put(gene2, list2);
                		}
            		}
            		
            	}
            }
            bufferedReader1.close();  
                
    } catch (Exception e) {
    	throw new IllegalArgumentException("Read saved model file error.");
    }
	}
	public void readNetworkFile(String filename){
		network = new HashMap<Integer, List<Integer>>();
		edgetype = new HashMap<Integer, List<String>>();
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
            		if(type.equalsIgnoreCase("regulator-target") && expressiondata.gene2int.get(gene2) != null) {
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
            				List<Integer> tgs = network.get(g1);
            				List<String> types = edgetype.get(g1);
            				if(!tgs.contains(g2)) {
            					tgs.add(g2);
            					types.add(type);
                				network.put(g1, tgs);
                				edgetype.put(g1, types);
            				}
            			} else {
            				List<Integer> tgs = new ArrayList<Integer>();
            				List<String> types = new ArrayList<String>();
            				tgs.add(g2);
            				types.add(type);
            				network.put(g1, tgs);
            				edgetype.put(g1, types);
            			}
            			if(network.get(g2) != null) {
            				List<Integer> tgs = network.get(g2);
            				List<String> types = edgetype.get(g2);
            				if(!tgs.contains(g1)) {
            					tgs.add(g1);
            					types.add(type);
                				network.put(g2, tgs);
                				edgetype.put(g2, types);
            				}
            			} else {
            				List<Integer> tgs = new ArrayList<Integer>();
            				List<String> types = new ArrayList<String>();
            				tgs.add(g1);
            				types.add(type);
            				network.put(g2, tgs);
            				edgetype.put(g2, types);
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
			List<Integer> neighbor;
			List<String> types;
			if(network.get(key) != null) {
				neighbor = network.get(key);
				types = edgetype.get(key);
			} else {
				neighbor = new ArrayList<Integer>();
				types = new ArrayList<String>();
			}
			List<String> tfs = tgtf.get(key);
			for(String tf:tfs) {
				List<Integer> tgs = tftg.get(tf);
				for(int tg:tgs) {
					if(!neighbor.contains(tg) && tg!=key) {
						neighbor.add(tg);
						types.add("co-regulated");
					}
				}
			}
			if(neighbor.size()>0) {
				network.put(key, neighbor);
				edgetype.put(key, types);
			}
		}
		
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
		System.out.println(sample.size());
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
			int returnVal = fc1.showOpenDialog(this);
			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc1.getSelectedFile();
				survivalField.setText(file.getAbsolutePath());
			}
		}  else if (esource == databutton) {
			int returnVal = fc1.showOpenDialog(this);
			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc1.getSelectedFile();
				dataField.setText(file.getAbsolutePath());
			}
		} else if (esource == modulebutton) {
			int returnVal = fc1.showOpenDialog(this);
			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc1.getSelectedFile();
				moduleField.setText(file.getAbsolutePath());
			}
		}  else if (esource == Run) {
			s1 = System.currentTimeMillis();
			szsurvival = survivalField.getText();
			szdataval = dataField.getText(); 
			szmoduleval = moduleField.getText(); 
			division = jpro.getValue().toString();
			
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
			String szMessage = "This is version 1.0.0 of Network-based Analysis.\n"
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
		JFrame frame = new Main_interface_singlegene();
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
