package bsm;


import bsm.core.*;

import java.awt.*;
import java.awt.event.*;
import java.io.*;

import javax.swing.*;
import javax.swing.border.TitledBorder;

import org.apache.commons.math3.distribution.NormalDistribution;

import java.util.*;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Class implementing the main input interface
 */
public class Main_interface extends JFrame implements ActionListener {
	
	static final String SZDELIM = "|;,";
	static final boolean BDEBUG = false;
	
	BSM_DataSet theDataSet;
	Integer[] PPIgene1;
	Integer[][][] PPIgene2;
	Double[][][] PPIvalue;
	List<RegulatorBindingData> KnownModules, Enrichmentfiles;
	List<PPIInteractionData> ppiData;
	int numbits;
	
	// Main
	static String szoptionalFile = " Optional";
	static String szDataFileDEF = " Required";
	static boolean bspotcheckDEF = false;
	static String szGeneAnnotationFileDEF = "";
	static String szCrossRefFileDEF = "";
	static int ndbDEF = 0;
	static int nstaticsourceDEF = 0;
	static int nummissing = 0;

	// Repeat
	static Vector<String> vRepeatFilesDEF0 = new Vector<String>();
	static Vector<String> vRepeatFilesDEF1 = new Vector<String>();
	ListDialog theRepeatList0, theRepeatList1;
	static Color lightBlue = new Color(190, 255, 190);
	Vector<String> knownmodulefiles, interactionfiles;

	// Search Options
	static double dCONVERGENCEDEF = 0.01;
	static double dMinScoreDEF = 0.0;
	static double dDELAYPATHDEF = .15;
	static double dDMERGEPATHDEF = .15;
	static double dPRUNEPATHDEF = .15;
	static double dTHRESHOLD = 0.000001;
	static double dPvalueshold = 0.01;

	// Filtering
	JRadioButton log1, log2;
	static int nnormalizeDEF = 1;
	ButtonGroup normGroup = new ButtonGroup();
	ButtonGroup goGroup = new ButtonGroup();
	ButtonGroup tfGroup = new ButtonGroup();
	ButtonGroup mirnaGroup = new ButtonGroup();
	
	String labels1[] = {" Min value", " Mean value", " Zero"};
	JComboBox comboBox1 = new JComboBox(labels1);
	//String labels4[] = {" Yes"," No"};
	//JComboBox comboBox4 = new JComboBox(labels4);
	JLabel filterlable;
	static int missinter = 0; // Set the missing value as
	static int nMaxMissingDEF = 0;

	// GUI
	static long s1;
	static boolean bendsearch = false;
	static String szexpressionval;
	static String szsavedval;
	HashMap<String, double[]> savemodule;
	HashMap<String, List<Integer>> savegenelist;
	HashMap<String, List<Integer>> saveposilist;
	HashMap<String, List<Integer>> savenegalist;
	static String missing;
	static boolean btakelog;
	static String epsilon;
	static String szmaxchild;
	static String szminchild;
	static String mincorr;
	static String DEthreshold;
	static String FDRthreshold;
	static String FCthreshold;

	int npathwaysourcecb;
	int ntfsourcecb;
	int nppisourcecb;
	String szuserFileField1;
	String szuserFileField2;
	String szuserFileField3;
	
	static JFileChooser theChooser = new JFileChooser();

	// Regulator Scoring GUI
	JButton regScoreHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JTextField regScoreField;
	static int NUMCOLS = 42;
	// Strings for the labels
	static Color gray = new Color(235,235,235);
	static Color defaultColor;
	static String[] staticsourceArray1 = { "User provided" };
	static String[] staticsourceArray2 = { "User provided" };
	static String[] staticsourceArray3 = { "User provided" };

	JTextField goField;
	JTextField extraField;
	JTextField xrefField;
	JTextField categoryIDField;
	JTextField taxonField;
	JTextField evidenceField;
	JButton infoButton = new JButton(Util.createImageIcon("About16.gif"));
	
	static JFileChooser fc0 = new JFileChooser(new File("Data/1_Regulator-target file"));
	static JFileChooser fc1 = new JFileChooser(new File("Data/2_Direct interaction file"));
	static JFileChooser fc3 = new JFileChooser();
	static JFileChooser fc4 = new JFileChooser();

	static Container contentPane;
	JPanel textpanel;
	JPanel panel4;

	JButton interactionbutton= new JButton();
	JButton knownmodulebutton= new JButton();
	JButton timeseriesbutton= new JButton();
	JButton savedmodelbutton= new JButton();
	JTextArea runningtext;
	JButton Run = new JButton();
	JButton currentButton = new JButton();
	JButton endSearchButton = new JButton();
	JTextField knownmoduleField, interactionField;
	JTextField expressionField;
	JTextField savedmodelField;
	final JSpinner j18, j19, j20, j21, j22;

	
	public Main_interface() throws FileNotFoundException, IOException {
		super("MulNet Module Detection v.1.0");

		contentPane = getContentPane();
		BoxLayout layout = new BoxLayout(contentPane, BoxLayout.Y_AXIS);
		contentPane.setLayout(layout);

		JPanel panel1=new JPanel();
		panel1.setBorder(new TitledBorder(null,"Load Data",TitledBorder.LEFT,TitledBorder.TOP));
		panel1.setLayout(new BorderLayout());
		
        /*****************************************************************/
		
		JLabel j5=new JLabel("Regulator-target file:");
		j5.setBounds(15,20,500,35);
		knownmoduleField=new JTextField(szoptionalFile, JLabel.TRAILING);
		knownmoduleField.setBounds(150,23,390,25);
		knownmoduleField.setBorder(BorderFactory.createLineBorder(Color.black));
		knownmoduleField.setOpaque(true);
		knownmoduleField.setBackground(Color.white);
		knownmodulebutton.setText("Load File");
		knownmodulebutton.setHideActionText(true);
		knownmodulebutton.addActionListener(this);
		knownmodulebutton.setBounds(545,23,90,25);
		
		JLabel j6=new JLabel("Direct interaction file:");
		j6.setBounds(15,50,500,35);
		interactionField=new JTextField(szoptionalFile, JLabel.TRAILING);
		interactionField.setBounds(150,53,390,25);
		interactionField.setBorder(BorderFactory.createLineBorder(Color.black));
		interactionField.setOpaque(true);
		interactionField.setBackground(Color.white);
		interactionbutton.setText("Load File");
		interactionbutton.setHideActionText(true);
		interactionbutton.addActionListener(this);
		interactionbutton.setBounds(545,53,90,25);
		
		/*****************************************************************/
		
		JLabel j7=new JLabel("Load expression file:");
		j7.setBounds(15,80,300,35);
		expressionField=new JTextField(szDataFileDEF, JLabel.TRAILING);
		expressionField.setBounds(150,83,390,25);
		expressionField.setBorder(BorderFactory.createLineBorder(Color.black));
		expressionField.setOpaque(true);
		expressionField.setBackground(Color.white);
		timeseriesbutton.setText("Load File");
		timeseriesbutton.setHideActionText(true);
		timeseriesbutton.addActionListener(this);
		timeseriesbutton.setBounds(545,83,90,25);
		
		/*****************************************************************/
		
		JLabel j9=new JLabel("Load saved modules:");
		j9.setBounds(15,110,300,35);
		savedmodelField=new JTextField(szoptionalFile, JLabel.TRAILING);
		savedmodelField.setBounds(150,113,390,25);
		savedmodelField.setBorder(BorderFactory.createLineBorder(Color.black));
		savedmodelField.setOpaque(true);
		savedmodelField.setBackground(Color.white);
		savedmodelbutton.setText("Load File");
		savedmodelbutton.setHideActionText(true);
		savedmodelbutton.addActionListener(this);
		savedmodelbutton.setBounds(545,113,90,25);
		
		/*****************************************************************/

		panel1.setLayout(null); 
		panel1.add(j5);
		panel1.add(j6);
		panel1.add(j7);
		panel1.add(j9);
		panel1.add(knownmoduleField);
		panel1.add(knownmodulebutton);
		panel1.add(interactionField);
		panel1.add(expressionField);
		panel1.add(interactionbutton);
		panel1.add(timeseriesbutton);
		panel1.add(savedmodelField);
		panel1.add(savedmodelbutton);
		
        JScrollPane   scrollpanel1   =   new   JScrollPane(panel1, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED, JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        scrollpanel1.setBounds(100, 100, 745, 80);
		panel1.setPreferredSize(new Dimension(scrollpanel1.getWidth() - 50, scrollpanel1.getHeight()*2));
		
		/*******************************************************/
		
		JPanel panel3=new JPanel();
		JScrollPane   scrollpanel2   =   new   JScrollPane(panel3, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED, JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        scrollpanel2.setBounds(100, 100, 745, 90);
        panel3.setPreferredSize(new Dimension(scrollpanel2.getWidth() - 50, scrollpanel2.getHeight()*2));
		panel3.setBorder(new TitledBorder(null,"Set Parameters",TitledBorder.LEFT,TitledBorder.TOP));
		panel3.setLayout(new BorderLayout());
		
		JPanel panel31=new JPanel();
		panel31.setBorder(new TitledBorder(null,"Data Preprocessing",TitledBorder.LEFT,TitledBorder.TOP));
		panel31.setBounds(5,20,365,150);
		panel31.setLayout(null);
		
		JLabel log=new JLabel("Log normalize data:");
		log.setBounds(15,20,300,35);
	    log1 = new JRadioButton("Yes");
		log2 = new JRadioButton("No");
		log1.setBounds(140, 25, 50, 22);
		log2.setBounds(190, 25, 50, 22);
		if (nnormalizeDEF == 0) {
			log1.setSelected(true);
		} else if (nnormalizeDEF == 1) {
			log2.setSelected(true);
		}
		
		JLabel mm=new JLabel("Max number of missing values:");
		mm.setBounds(15,55,300,35);
		SpinnerNumberModel misscontrol = new SpinnerNumberModel(new Integer(nummissing), new Integer(0), null, new Integer(1));
		j18 = new JSpinner(misscontrol);
		j18.setBounds(210,60,45,22);
		
		JLabel inter=new JLabel("Set missing values as:");
		inter.setBounds(15,95,300,35);
		comboBox1.setBounds(155,100,100,21);
	    if (missinter == 0) {
	    	comboBox1.setSelectedIndex(0);
		} else if(missinter == 1){
			comboBox1.setSelectedIndex(1);
		} else{
			comboBox1.setSelectedIndex(2);
		}
		
		panel31.add(log);
		panel31.add(log1);
		panel31.add(log2);
		normGroup.add(log1);
		normGroup.add(log2);
		panel31.add(mm);
		panel31.add(j18);
		panel31.add(inter);
		panel31.add(comboBox1);
		
		/*******************************************************/
	    JPanel panel32=new JPanel();
		panel32.setBorder(new TitledBorder(null,"Module Detecting",TitledBorder.LEFT,TitledBorder.TOP));
		panel32.setBounds(370,20,365,150);
		panel32.setLayout(null);

		JLabel minir=new JLabel("Min correlation coefficient:");
		minir.setBounds(15,20,300,35);
		SpinnerNumberModel minipearson = new SpinnerNumberModel(0.4, -1, 1, 0.1);
		j22 = new JSpinner(minipearson);
		j22.setBounds(175,25,50,22);
		
		JLabel knn=new JLabel("Discount factor for the MDP:");
		knn.setBounds(15,55,200,35);
		SpinnerNumberModel con = new SpinnerNumberModel(0.995, 0, 1, 0.001);
		j20 = new JSpinner(con);
		j20.setBounds(185,60,50,22);
		
		JLabel nummodule1=new JLabel("Max");
		nummodule1.setBounds(15,90,30,35);
		SpinnerNumberModel converenge = new SpinnerNumberModel(50, 2, null, 1);
		j21 = new JSpinner(converenge);
		j21.setBounds(42,95,50,22);
		JLabel nummodule2=new JLabel("and min");
		nummodule2.setBounds(95,90,50,35);	
		SpinnerNumberModel minigene = new SpinnerNumberModel(1, 1, null, 1);
		j19 = new JSpinner(minigene);
		j19.setBounds(145,95,50,22);	
		JLabel nummodule3=new JLabel("number of modules");
		nummodule3.setBounds(198,90,200,35);

		panel32.add(knn);
		panel32.add(nummodule1);
		panel32.add(nummodule2);
		panel32.add(nummodule3);
		panel32.add(minir);
		panel32.add(j19);
		panel32.add(j20);
		panel32.add(j21);
		panel32.add(j22);

		panel3.setLayout(null);
		panel3.add(panel31); 
		panel3.add(panel32);
		/********************************************************/
		textpanel=new JPanel();
		textpanel.setPreferredSize(new Dimension(745, 150));
		textpanel.setBorder(new TitledBorder(null,"Search Gene Modules",TitledBorder.LEFT,TitledBorder.TOP));
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
		endSearchButton.setText("Stop Searching");
		endSearchButton.setEnabled(false);
		endSearchButton.addActionListener(this);
		infoButton.addActionListener(this);
		infoButton.setBackground(gray);
		
		JPanel panel6=new JPanel();
		panel6.setPreferredSize(new Dimension(745, 50));
		panel6.add(Run);
		panel6.add(endSearchButton);
		panel6.add(infoButton);
		contentPane.add(scrollpanel1);
		contentPane.add(scrollpanel2);
		contentPane.add(textpanel);
		contentPane.add(panel6);
		
		theRepeatList0 = new ListDialog(this, Main_interface.vRepeatFilesDEF0,
				knownmodulebutton, Main_interface.fc0);
		theRepeatList1 = new ListDialog(this, Main_interface.vRepeatFilesDEF1,
				interactionbutton, Main_interface.fc1);
	}

	private BSM_DataSet buildset(String szexp1val, 
			Integer maxcase, Integer missinter, boolean btakelog) 
			throws Exception {
		BSM_DataSet theDataSet1 = new BSM_DataSet(szexp1val, maxcase, missinter, btakelog);
		theDataSet1 = new BSM_DataSet(theDataSet1.filterMissing());
		theDataSet1 = new BSM_DataSet(theDataSet1.maxAndFilterDuplicates());
		return theDataSet1;	
	}
	
	/**
	 * A control method that handles the response for when the execute button on
	 * the interface is pressed including building the data set, running the
	 * TSMiner modeling procedure, and displaying the results
	 */
	public void clusterscript(Vector<String> knownmodules, Vector<String> ppifile,
			String szexp1val, String szsavedval, String missing, int missinter, boolean btakelog) throws Exception {
		
		//szexp1val = "D:\\Personal issues\\MulNet\\tool\\MulNet v.2.0\\4_Expression data\\Benchamrk_Data\\Single-cell RNA-seq/Processed_Ovarian.txt";
		//String save1 = "D:\\Personal issues\\MulNet\\tool\\MulNet v.2.0\\Data\\1_Regulator-target file/human TF-target.txt";
		//String save2 = "D:\\Personal issues\\MulNet\\tool\\MulNet v.2.0\\Data\\1_Regulator-target file/human miRNA-target.txt";
		//String ppi = "D:\\Personal issues\\MulNet\\tool\\MulNet v.2.0\\Data\\2_Direct interaction file/human PPI.txt";
		//szsavedval = "D:\\Personal issues\\MulNet 2.0\\result\\1 module detection\\0.6-0.998/primary Fibroblast.txt";
		//knownmodules.add(save1);
		//knownmodules.add(save2);
		//ppifile.add(ppi);
		
				if (szexp1val.trim().equals("")) {
					runningtext.append("No expression data file given.");
					runningtext.paintImmediately(runningtext.getBounds());
					throw new IllegalArgumentException("No expression data file given!");
				} else if (!(new File(szexp1val)).exists()) {
					runningtext.append("The expression data file '" + szexp1val+ "' cannot be found.");
					runningtext.paintImmediately(runningtext.getBounds());
					throw new IllegalArgumentException("The expression data file '" + szexp1val+ "' cannot be found.");
				}
				
				bendsearch = false;
				theDataSet = buildset(szexp1val, Integer.parseInt(missing), missinter, btakelog);
				theDataSet.normalization();
				List<String> genelist = new ArrayList<String>();
				for(int i=0;i<theDataSet.numrows;i++) {
					genelist.add(theDataSet.genenames[i]);
				}
				System.out.println(genelist.size());
				/*
				try {
					BufferedWriter outXml1 = new BufferedWriter(new FileWriter("D:/expression.txt"));
					outXml1.write("Name"+"\t");
					for(int i=0;i<theDataSet.numcols;i++) {
						outXml1.write(theDataSet.dsamplemins[i]+"\t");
					}
					outXml1.newLine();
					for(int i=0;i<theDataSet.numrows;i++) {
						outXml1.write(theDataSet.genenames[i]+"\t");
						for(int j=0;j<theDataSet.numcols;j++) {
							outXml1.write(theDataSet.controlnorm[i][j]+"\t");
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
				*/
				
				GetCorrMatrix gcm = new GetCorrMatrix();
				gcm.multithread(theDataSet.controlnorm, theDataSet.numrows, 500);
				
				double NDmean = GetCorrMatrix.Sumcorr / GetCorrMatrix.Suncount;
				double dval = GetCorrMatrix.Suncount / (GetCorrMatrix.Suncount - 1)
						* (GetCorrMatrix.Sumcorrsq / GetCorrMatrix.Suncount - Math.pow(NDmean, 2));
				double NDsigma = Math.sqrt(dval);
				System.out.println(NDmean+"\t"+NDsigma);
				NormalDistribution initNormal = new NormalDistribution(NDmean, NDsigma);
				
				HashMap<Integer, List<List<Integer>>> ppiindex = new HashMap<Integer, List<List<Integer>>>();
				HashMap<Integer, List<Double[]>> ppivalue = new HashMap<Integer, List<Double[]>>();
				
				runningtext.append("Read interaction data..."+"\n");
				runningtext.paintImmediately(runningtext.getBounds());
				if(knownmodules!=null && knownmodules.size()>0) {
					KnownModules = new ArrayList<RegulatorBindingData>();
						for(String file:knownmodules) {
							RegulatorBindingData pathwayData = new RegulatorBindingData(file,
									genelist, SZDELIM, null, null);
							pathwayData.deDuplication();
							pathwayData.coexpression(theDataSet, initNormal, 
							Double.parseDouble(mincorr), ppiindex, ppivalue);
							KnownModules.add(pathwayData);
						}
				}
				
				if(ppifile != null && ppifile.size() > 0) {
					ppiData = new ArrayList<PPIInteractionData>();
					try {
						for(String file:ppifile) {
							PPIInteractionData ppidata = new PPIInteractionData(file, genelist);
							ppidata.coexpression(theDataSet, initNormal, 
									Double.parseDouble(mincorr), ppiindex, ppivalue);
							ppiData.add(ppidata);
						}
					} catch (Exception e) {
						runningtext.append("Read file "+ ppifile +" error.");
						runningtext.paintImmediately(runningtext.getBounds());
			            throw new IllegalArgumentException("Read file "+ ppifile +" error.");
			        }
				}

				if(ppiindex.size()>0) {
					PPIgene1 = new Integer[ppiindex.size()];
					PPIgene2 = new Integer[ppiindex.size()][][];
					PPIvalue = new Double[ppiindex.size()][][];
					int count=0;
					Collection<Integer> cl = ppiindex.keySet();
					Iterator<Integer> itr = cl.iterator();
					while (itr.hasNext()) {
					    int key = itr.next();
					    List<List<Integer>> index = ppiindex.get(key);
					    List<Double[]> value = ppivalue.get(key);
					    PPIgene1[count] = key;
					    PPIgene2[count] = new Integer[index.size()][];
					    PPIvalue[count] = new Double[index.size()][];
					    for(int i=0;i<index.size();i++) {	    		
					    	PPIgene2[count][i] = (index.get(i)).toArray(new Integer[index.get(i).size()]);
					    	PPIvalue[count][i] = value.get(i);
					    }
					    count++;
					}
				}
				
				File directory = new File("Data/3_Enrichment analysis file");
				File[] enrichment = directory.listFiles();
				Enrichmentfiles = new ArrayList<RegulatorBindingData>();
				if(enrichment!=null && enrichment.length>0) {
					for(File file:enrichment) {
						if(file.exists()) {
							RegulatorBindingData pathwayData = new RegulatorBindingData(file.toString(),
									genelist, SZDELIM, null, null);
							pathwayData.deDuplication();
							Enrichmentfiles.add(pathwayData);
						}
					}
				} else {
					runningtext.append("No file in the \"3_Enrichment analysis file\" folder."+"\n");
					runningtext.paintImmediately(runningtext.getBounds());
				}
				
				if(szsavedval != null && szsavedval.length()>0 && !szsavedval.equals(" Optional")){
				
				File file = new File(szsavedval);
				if(file.exists()) {
					readFile(szsavedval);
					if(savemodule.size()==0) {
						runningtext.append("Please check the saved model file."+"\n");
						runningtext.paintImmediately(runningtext.getBounds());
						throw new IllegalArgumentException("Please check the saved model file.");
					}
				} else {
					runningtext.append("Cannot find the saved model file: "+szsavedval+"\n");
					runningtext.paintImmediately(runningtext.getBounds());
					throw new IllegalArgumentException("Cannot find the saved model file: "+szsavedval+"\n");
				}
					
				}
				/******************************************Module detection***********************************************/		
		        LearningCase caseModules = new LearningCase(theDataSet, PPIgene1, PPIgene2, PPIvalue, 
		        		Enrichmentfiles, savemodule, savegenelist, saveposilist, savenegalist, szmaxchild,
		        		szminchild, mincorr, epsilon, runningtext, endSearchButton);

		        final LearningCase finalcase = caseModules;
		        /*******************************************Network**********************************************/
		        for(int i=0; i<finalcase.treeptr.numchildren; i++) {
			    	GetNetwork network = new GetNetwork();
					network.network(finalcase.treeptr.nextptr[i], theDataSet, KnownModules, ppiData, Double.parseDouble(mincorr));
		        }
		        
				/*******************************************Display**********************************************/
		        try {
		        	javax.swing.SwingUtilities.invokeLater(new Runnable() {
		        		public void run() {
		        			JFrame frame1 = new JFrame("Resulting Modules");
							Container theDialogContainer = frame1.getContentPane();
							theDialogContainer.setBackground(Color.white);
							JTabbedPane tabbedPane = new JTabbedPane();
								CaseTable newContentPane1 = new CaseTable(frame1, theDataSet, finalcase, 
										finalcase.treeptr);
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
		        } catch (Exception e) {
		        	e.printStackTrace();
				}
				
		
		long e1 = System.currentTimeMillis();
		System.out.println("Time: " + (e1 - s1) + "ms");
	}
	
	
	public void readFile(String filename){
		savemodule = new HashMap<String, double[]>();
		savegenelist = new HashMap<String, List<Integer>>();
		saveposilist = new HashMap<String, List<Integer>>();
		savenegalist = new HashMap<String, List<Integer>>();
		try {
			BufferedReader bufferedReader1 = new BufferedReader(new FileReader(filename));   
			String lineTxt = bufferedReader1.readLine();
			String[] header = lineTxt.split("\t");
			
			if(header.length > 2){
				while((lineTxt = bufferedReader1.readLine()) != null){
					String[] value = lineTxt.split("\t");
					if(theDataSet.gene2int.get(value[0].replace("\"", "").replace(" ", "")) != null) {
						int gene = theDataSet.gene2int.get(value[0].replace("\"", "").replace(" ", ""));
						//System.out.println(gene);
						String module = value[1];
						int cor = Integer.parseInt(value[2].replace("\"", "").replace(" ", ""));
						if(savemodule.get(module) != null) {
							double[] row = savemodule.get(module);
							double[] expr = theDataSet.controlnorm[gene];
							if(cor>0) {						
								for(int i=0;i<row.length;i++) {
									row[i] += expr[i];
								}
								savemodule.put(module, row);
							} else if(cor<0) {
								for(int i=0;i<row.length;i++) {
									row[i] -= expr[i];
								}
								savemodule.put(module, row);
							}
						} else {
							double[] row = new double[theDataSet.numcols];
							double[] expr = theDataSet.controlnorm[gene];
							if(cor>0) {
								for(int i=0;i<row.length;i++) {
									row[i] = expr[i];
								}
								savemodule.put(module, row);
							} else if(cor<0) {
								for(int i=0;i<row.length;i++) {
									row[i] = -expr[i];
								}
								savemodule.put(module, row);
							}
						}
						if(savegenelist.get(module) != null) {
							List<Integer> genelist = savegenelist.get(module);
							genelist.add(gene);
							savegenelist.put(module, genelist);
						} else {
							List<Integer> genelist = new ArrayList<Integer>();
							genelist.add(gene);
							savegenelist.put(module, genelist);
						}
						if(cor>0) {
							if(saveposilist.get(module) != null) {
								List<Integer> posilist = saveposilist.get(module);
								posilist.add(gene);
								saveposilist.put(module, posilist);
							} else {
								List<Integer> posilist = new ArrayList<Integer>();
								posilist.add(gene);
								saveposilist.put(module, posilist);
							}
						}
						if(cor<0) {
							if(savenegalist.get(module) != null) {
								List<Integer> negalist = savenegalist.get(module);
								negalist.add(gene);
								savenegalist.put(module, negalist);
							} else {
								List<Integer> negalist = new ArrayList<Integer>();
								negalist.add(gene);
								savenegalist.put(module, negalist);
							}
						}
					}
                }
				HashMap<String, double[]> savemodule2 = new HashMap<String, double[]>();
				Collection<String> cl = savemodule.keySet();
				Iterator<String> itr = cl.iterator();
				while (itr.hasNext()) {
					String module = itr.next();
					double[] row = savemodule.get(module);
					List<Integer> list = savegenelist.get(module);
					for(int i=0;i<row.length;i++) {
						row[i] /= list.size();
					}
					savemodule2.put(module, row);
				}
				savemodule = savemodule2;
			} else {
        		runningtext.append("The saved module file should have at least 3 columns.");
        		runningtext.paintImmediately(runningtext.getBounds());
        		bufferedReader1.close();  
            	throw new IllegalArgumentException("The saved module file should have at least 3 columns.");
        	}
            bufferedReader1.close();  
                
    } catch (Exception e) {
    	runningtext.append("Read saved model file error.");
		runningtext.paintImmediately(runningtext.getBounds());
    	throw new IllegalArgumentException("Read saved model file error.");
    }
	}

	public boolean judgeContainsStr(String cardNum) {
    	String regex=".*[a-zA-Z]+.*";
	    Matcher m=Pattern.compile(regex).matcher(cardNum);
	    return m.matches();
    }
	public boolean isNumber(String string) {
        if (string == null) return false;
        Pattern pattern = Pattern.compile("^-?\\d+(\\.\\d+)?$");
        return pattern.matcher(string).matches();
    }
	/**
	 * define the button methods
	 */
	public void actionPerformed(ActionEvent e) {
		Object esource = e.getSource();

		if (esource == endSearchButton) {
			bendsearch = true;
			runningtext.append("End Search Requested. Search Will End Soon..."+"\n");
			runningtext.paintImmediately(runningtext.getBounds());
			endSearchButton.setEnabled(false);
		} else if (esource == knownmodulebutton) {
			theRepeatList0.setLocation(this.getX() + 75, this.getY() + 100);
			theRepeatList0.setVisible(true);
		} else if (esource == interactionbutton) {
			theRepeatList1.setLocation(this.getX() + 75, this.getY() + 100);
			theRepeatList1.setVisible(true);
		} else if (esource == timeseriesbutton) {
			int returnVal = fc3.showOpenDialog(this);
			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc3.getSelectedFile();
				expressionField.setText(file.getAbsolutePath());
			}
		} else if (esource == savedmodelbutton) {
			int returnVal = fc4.showOpenDialog(this);
			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc4.getSelectedFile();
				savedmodelField.setText(file.getAbsolutePath());
			}
		}  else if (esource == Run) {
			s1 = System.currentTimeMillis();
			
			knownmodulefiles = theRepeatList0.data;
			interactionfiles = theRepeatList1.data;
			szexpressionval = expressionField.getText(); 
			szsavedval = savedmodelField.getText(); 
			missing = j18.getValue().toString();
			epsilon = j20.getValue().toString();
			szmaxchild = j21.getValue().toString();
			szminchild = j19.getValue().toString();
			mincorr = j22.getValue().toString();
			
			int temp2=comboBox1.getSelectedIndex();
			if(temp2==0){
				missinter=0;
			}else if(temp2==1){
				missinter=1;
			}else{
				missinter=2;
			}
			
			btakelog = log1.isSelected();

			this.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));

			final JFrame fframe = this;

			Runnable clusterrun = new Runnable() {
				public void run() {
					Run.setEnabled(false);
					try {
						clusterscript(knownmodulefiles, interactionfiles, szexpressionval, 
								szsavedval, missing, missinter, btakelog);
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
			String szMessage = "This is version 1.0.0 of MulNet Module Detection.\n\n"
					+ "The MulNet is available under a GPL v3.0 license.\n"
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
