package bsm;

import bsm.core.*;

import javax.swing.*;
import org.apache.commons.math3.distribution.NormalDistribution;
import java.util.*;
import java.text.*;
import java.io.*;

/**
 * This class implements the core methods for learning the BSM models
 */

public class LearningCase {
	static final boolean BDEBUG = false;
	static final boolean BDEBUGMODEL = false;
	boolean bhasmerge = false;
	int MinNum = 10;
	int minnum = 5;
	int npath;
	int ninitsearchval;
	int MinGeneNum;
	JButton currentButton;
	JTextArea casetext;
	static final String SZDELIM = "|;,";
	double[] dprevouterbestlog;
	double BEPSILON = 0.00001;
	double EPSILON = 0.99;
	int nmaxchild;
	int nminchild;
	double minPearson = 0.4;
	int numrows1, numrows2, numcols;
	int MINPATH = 5;
	int ntrain, ntest;
	double[][] traindata;
	Random theRandom;
	static final double RIDGE = 1;
	/**
	 * Object containing all of the Regulator-Gene binding data.
	 */
	Integer[] PPIgene1;
	Integer[][][] PPIgene2;
	Double[][][] PPIvalue;
	List<RegulatorBindingData> pathwaydata;
	double NDmean;
	double NDsigma;
	NormalDistribution Normalcorr;
	double Sumcorr;   
	double Sumcorrsq;
	Casenode treeptr;
	DecimalFormat df2;
	double dbestlog;
	double peaklike;
	double currentbestlike;
	Casenode bestTree = null;
	BSM_DataSet theDataSet; 
	int numtotalPath = 1;
	String[][] resultList;
	List<String[][]> minenrichmenttable;
	boolean[] IsSeed;
	
	public LearningCase(BSM_DataSet theDS, Integer[] PPIgene1, Integer[][][] PPIgene2, Double[][][] PPIvalue, 
			List<RegulatorBindingData> pathwaydata, HashMap<String, double[]> savedmodule, HashMap<String, List<Integer>> savegenelist,
			HashMap<String, List<Integer>> saveposilist, HashMap<String, List<Integer>> savenegalist, String szmaxchild, 
			String szminchild, String Pearson1, String epsilon, JTextArea casetext, JButton endSearchButton) throws Exception {
	
		df2 = new DecimalFormat("#0.0000");
		this.theDataSet = theDS;
		this.PPIgene1 = PPIgene1;
		this.PPIgene2 = PPIgene2;
		this.PPIvalue = PPIvalue;
		this.nmaxchild = Integer.parseInt(szmaxchild);
		this.nminchild = Integer.parseInt(szminchild);
		this.pathwaydata = pathwaydata;
		this.casetext = casetext;
		this.minPearson = Double.parseDouble(Pearson1);
		this.EPSILON = Double.parseDouble(epsilon);
		this.numrows1 = theDataSet.numrows;
		this.numcols = theDataSet.numcols;
		treeptr = new Casenode(); //Node Initialization
		theRandom = new Random();
		
		if(PPIgene1 != null && PPIgene1.length > 0) {
			this.numrows2 = PPIgene1.length;
		}
	    casetext.append(" Gene: "+numrows1+"\t"+" Functional unit: "+numrows2+"\t"+" Sample: "+numcols+"\n");
	    casetext.paintImmediately(casetext.getBounds());
		
	    if(savedmodule != null && savedmodule.size()>0){
	    	traindata = theDataSet.controlnorm;
	    	System.out.println("Module number: " + savedmodule.size());
		    generateTree(treeptr, savedmodule, savegenelist, saveposilist, savenegalist);
		    //trainhmm2(treeptr);
		    //softclustering2(treeptr);
		    
		    EnrichmentAnalysis EA = new EnrichmentAnalysis();
		    if(pathwaydata != null && pathwaydata.size() > 0){
		    	minenrichmenttable = new ArrayList<String[][]>();
		    	for(int i=0;i<pathwaydata.size();i++) {
		    		RegulatorBindingData knownmodule = pathwaydata.get(i);
		    		for(int nchild=0;nchild<treeptr.numchildren;nchild++){
			    		List<Integer> posilist = treeptr.nextptr[nchild].posi_genelist;
						List<Integer> negalist = treeptr.nextptr[nchild].nega_genelist;
						treeptr.nextptr[nchild].enrichment.add(EA.enrichment(posilist, negalist,
								knownmodule.regNames, knownmodule.reg2GeneBindingIndex, numrows1));
			    	}
		    		String[][] mintable = new String[knownmodule.regNames.length+1][4];
		    		mintable[0][0] = "Name";
		    		mintable[0][1] = "Total";
		    		mintable[0][2] = "Overlap";
		    		mintable[0][3] = "Pvalue";
			    	for(int j=0;j<knownmodule.regNames.length;j++){
			    		double minpvalue = 1;
			    		int count = 0;
			    		double overlap = 0; 
			    		for(int nchild=0;nchild<treeptr.numchildren;nchild++){
				    		String[][] table = treeptr.nextptr[nchild].enrichment.get(i);
				    		double both = Double.parseDouble(table[j+1][7]);
				    		if(minpvalue > both){
				    			minpvalue = both;
				    			count = Integer.parseInt(table[j+1][1]);
					    		overlap = Integer.parseInt(table[j+1][6]);
				    		}
				    	}
			    		mintable[j+1][0] = knownmodule.regNames[j];
			    		mintable[j+1][1] = count+"";
			    		mintable[j+1][2] = overlap+"";
			    		mintable[j+1][3] = minpvalue+"";
			    	}
			    	int count1=0;
			    	int count2=0;
			    	for(int j=1;j<mintable.length;j++) {
			    		if(Double.parseDouble(mintable[j][3])<=0.01) {
			    			count1++;
			    			count2++;
			    		}else if(Double.parseDouble(mintable[j][3])<=0.05) {
			    			count2++;
			    		}
			    	}
			    	System.out.println("Significant:"+"\t"+count1+"\t"+count2);
			    	minenrichmenttable.add(mintable);
		    	}
		    }
		    
		    try{
				BufferedWriter outXml = new BufferedWriter(new FileWriter("Final modules.txt"));
				outXml.write("Gene"+"\t"+"Module"+"\t"+"Correlation"+"\t");
				for(int i=0;i<theDataSet.numcols-1;i++) {
					outXml.write(theDataSet.dsamplemins[i]+"\t");
				}
				outXml.write(theDataSet.dsamplemins[theDataSet.numcols-1]+"\n");
				
				for(int i=0;i<treeptr.numchildren;i++){
					List<Integer> posigenelist = treeptr.nextptr[i].posi_genelist;
					List<Integer> negagenelist = treeptr.nextptr[i].nega_genelist;
					if(posigenelist.size()>0){
						for(int j=0;j<posigenelist.size();j++){
							int geneindex = posigenelist.get(j);
					    	outXml.write(theDataSet.genenames[geneindex]+"\t"+"Module_"+(i+1)+"\t"+"1"+"\t");
					    	for(int m=0;m<theDataSet.numcols-1;m++) {
					    		outXml.write(theDataSet.controldata[geneindex][m]+"\t");
					    	}
					    	outXml.write(theDataSet.controldata[geneindex][theDataSet.numcols-1]+"\n");
					    }
					}
					if(negagenelist.size()>0){
						for(int j=0;j<negagenelist.size();j++){
							int geneindex = negagenelist.get(j);
					    	outXml.write(theDataSet.genenames[geneindex]+"\t"+"Module_"+(i+1)+"\t"+"-1"+"\t");
					    	for(int m=0;m<theDataSet.numcols-1;m++) {
					    		outXml.write(theDataSet.controldata[geneindex][m]+"\t");
					    	}
					    	outXml.write(theDataSet.controldata[geneindex][theDataSet.numcols-1]+"\n");
					    }
					}
				}
				outXml.flush(); 
				outXml.close();
				System.out.println("DONE");
			}catch (Exception e) {
				System.out.println("FALSE"); 
			e.printStackTrace(); 
			}
	    } else {
	    	traindata = theDataSet.controlnorm;
	    	List<Integer> genelist = new ArrayList<Integer>();
		    for(int i=0;i<traindata.length;i++) genelist.add(i);
		    BuildSSM ssm = new BuildSSM();
		    double[] curve = ssm.meancurve(traindata, genelist);
		    buildEmptyTree(treeptr, curve);
		    endSearchButton.setEnabled(true);
		    
		    searchstage1(true);
		    trainhmm(treeptr, 200);

		    if (endSearchButton != null) {
				endSearchButton.setEnabled(false);
			}
			casetext.append("================================"+"\n");
			casetext.paintImmediately(casetext.getBounds());
			
			System.out.println("Final path = " + treeptr.numchildren);
			
		    softclustering2(treeptr);
		    EnrichmentAnalysis EA = new EnrichmentAnalysis();
		    if(pathwaydata != null && pathwaydata.size() > 0){
		    	minenrichmenttable = new ArrayList<String[][]>();
		    	for(int i=0;i<pathwaydata.size();i++) {
		    		RegulatorBindingData knownmodule = pathwaydata.get(i);
		    		for(int nchild=0;nchild<treeptr.numchildren;nchild++){
			    		List<Integer> posilist = treeptr.nextptr[nchild].posi_genelist;
						List<Integer> negalist = treeptr.nextptr[nchild].nega_genelist;
						treeptr.nextptr[nchild].enrichment.add(EA.enrichment(posilist, negalist,
								knownmodule.regNames, knownmodule.reg2GeneBindingIndex, numrows1));
			    	}
		    		String[][] mintable = new String[knownmodule.regNames.length+1][4];
		    		mintable[0][0] = "Name";
		    		mintable[0][1] = "Total";
		    		mintable[0][2] = "Overlap";
		    		mintable[0][3] = "Pvalue";
			    	for(int j=0;j<knownmodule.regNames.length;j++){
			    		double minpvalue = 1;
			    		int count = 0;
			    		double overlap = 0; 
			    		for(int nchild=0;nchild<treeptr.numchildren;nchild++){
				    		String[][] table = treeptr.nextptr[nchild].enrichment.get(i);
				    		double both = Double.parseDouble(table[j+1][7]);
				    		if(minpvalue > both){
				    			minpvalue = both;
				    			count = Integer.parseInt(table[j+1][1]);
					    		overlap = Integer.parseInt(table[j+1][6]);
				    		}
				    	}
			    		mintable[j+1][0] = knownmodule.regNames[j];
			    		mintable[j+1][1] = count+"";
			    		mintable[j+1][2] = overlap+"";
			    		mintable[j+1][3] = minpvalue+"";
			    	}
			    	minenrichmenttable.add(mintable);
		    	}
		    }
		    
		    try{
				BufferedWriter outXml = new BufferedWriter(new FileWriter("Final modules.txt"));
				outXml.write("Gene"+"\t"+"Module"+"\t"+"Correlation"+"\t");
				for(int i=0;i<theDataSet.numcols-1;i++) {
					outXml.write(theDataSet.dsamplemins[i]+"\t");
				}
				outXml.write(theDataSet.dsamplemins[theDataSet.numcols-1]+"\n");
				
				for(int i=0;i<treeptr.numchildren;i++){
					List<Integer> posigenelist = treeptr.nextptr[i].posi_genelist;
					List<Integer> negagenelist = treeptr.nextptr[i].nega_genelist;
					if(posigenelist.size()>0){
						for(int j=0;j<posigenelist.size();j++){
							int geneindex = posigenelist.get(j);
					    	outXml.write(theDataSet.genenames[geneindex]+"\t"+"Module_"+(i+1)+"\t"+"1"+"\t");
					    	for(int m=0;m<theDataSet.numcols-1;m++) {
					    		outXml.write(theDataSet.controldata[geneindex][m]+"\t");
					    	}
					    	outXml.write(theDataSet.controldata[geneindex][theDataSet.numcols-1]+"\n");
					    }
					}
					if(negagenelist.size()>0){
						for(int j=0;j<negagenelist.size();j++){
							int geneindex = negagenelist.get(j);
					    	outXml.write(theDataSet.genenames[geneindex]+"\t"+"Module_"+(i+1)+"\t"+"-1"+"\t");
					    	for(int m=0;m<theDataSet.numcols-1;m++) {
					    		outXml.write(theDataSet.controldata[geneindex][m]+"\t");
					    	}
					    	outXml.write(theDataSet.controldata[geneindex][theDataSet.numcols-1]+"\n");
					    }
					}
				}
				outXml.flush(); 
				outXml.close();
				System.out.println("DONE");
			} catch (Exception e) {
				System.out.println("FALSE"); 
				e.printStackTrace(); 
			}
	    }
	    
	    /*
		boolean bagain; //Delete modules with lower than 5 genes
			do {
				bagain = false;
				softclustering2(treeptr);
				int theMinPathRec = findMinPath(treeptr);		
				if (MinGeneNum < MINPATH) {//鍒犻櫎5涓熀鍥犱互涓嬬殑璺緞
					casetext.append(" Module "+theMinPathRec+" are filtered for including lower than 5 genes."+"\n");
					casetext.paintImmediately(casetext.getBounds());
					deleteMinPath(theMinPathRec, treeptr); //鍒犻櫎鍖呭惈鏈�灏戝熀鍥犵殑璺緞
					bagain = true;
					trainhmm(treeptr, 200);
					if (BDEBUG) {
						System.out.println("after retrain");
					}
					numtotalPath--;
				}
			} while (bagain);
			*/
			
			resultList = new String[treeptr.numchildren][numcols+4];
		    for(int i=0;i<treeptr.numchildren;i++){
				resultList[i][0] = i+1+"";
				resultList[i][1] = treeptr.nextptr[i].genelist.size()+"";
				resultList[i][2] = treeptr.nextptr[i].posi_genelist.size()+"";
				resultList[i][3] = treeptr.nextptr[i].nega_genelist.size()+"";
				double[] mean = treeptr.nextptr[i].curve;
				for(int j=0;j<mean.length;j++){
					resultList[i][j+4] = mean[j]+"";
			    }
			}
			
			casetext.append("================================"+"\n");
			casetext.paintImmediately(casetext.getBounds());	
	}

	private void generateTree(Casenode ptr, HashMap<String, double[]> savedmodule, HashMap<String, List<Integer>> savegenelist,
			HashMap<String, List<Integer>> saveposilist, HashMap<String, List<Integer>> savenegalist){
		ptr.numchildren = savedmodule.size();
		int count = 0;
		Collection<String> cl = savedmodule.keySet();
		Iterator<String> itr = cl.iterator();
		while (itr.hasNext()) {
			String module = itr.next();
			ptr.nextptr[count] = new Casenode(ptr);
			ptr.nextptr[count].curve = savedmodule.get(module);
			if(savegenelist.get(module) != null) {
				ptr.nextptr[count].genelist = savegenelist.get(module);
			}
			if(saveposilist.get(module) != null) {
				ptr.nextptr[count].posi_genelist = saveposilist.get(module);
			}
			if(savenegalist.get(module) != null) {
				ptr.nextptr[count].nega_genelist = savenegalist.get(module);
			}
			count++;
		}
	}
	
	/**
	 * First step of tree searching锛宎dd and delete path
	 */
	public void searchstage1(boolean firsttrain) throws Exception {
		dbestlog = trainhmm(treeptr, 200);
		dprevouterbestlog = new double[nmaxchild];
		dprevouterbestlog[numtotalPath-1] = dbestlog;
		boolean bendsearchlocal;
		Casenode finalTree = (Casenode) treeptr.clone();
		peaklike = 0;
		boolean keepon;
		do {
			bestTree = treeptr;
			keepon = false;
			currentbestlike = 0;
			/************************add path****************************/
			double avgcor = hardclustering(treeptr);
			double traincor = Math.max(0.4, Math.min(avgcor, 0.8));
			
			for (int nchild = 0; nchild < treeptr.numchildren; nchild++){
				List<Integer> posigenes = treeptr.nextptr[nchild].posi_genelist;	
				if(posigenes.size() >= MinNum){
					List<Integer> posilist = new ArrayList<Integer>();
					List<Integer> restlist = new ArrayList<Integer>();
					for(int gene:posigenes){
						double r = treeptr.nextptr[nchild].dcorrec[gene];
						if(Math.abs(r) < traincor){
							posilist.add(gene);
						} else {
							restlist.add(gene);
						}
					}
					//for(int gene:negagenes){
					//	restlist.add(gene);
					//}
					if(posilist.size() >= minnum && restlist.size() >= minnum){
						traverseandadd(treeptr, nchild, posilist, restlist);
					}
				}
				List<Integer> negagenes = treeptr.nextptr[nchild].nega_genelist;
				if(negagenes.size() >= MinNum){
					List<Integer> negalist = new ArrayList<Integer>();
					List<Integer> restlist = new ArrayList<Integer>();
					for(int gene:negagenes){
						double r = treeptr.nextptr[nchild].dcorrec[gene];
						if(Math.abs(r) < traincor){
							negalist.add(gene);
						} else {
							restlist.add(gene);
						}
					}
					//for(int gene:posigenes){
					//	restlist.add(gene);
					//}
					if(negalist.size() >= minnum && restlist.size() >= minnum){
						traverseandadd(treeptr, nchild, negalist, restlist);
					}
				}
			}
			/****************************************************/

			if(currentbestlike > 0){
				treeptr = bestTree;
				numtotalPath++;
				dprevouterbestlog[numtotalPath-1] = currentbestlike;
				boolean delete = traverseanddelete(treeptr);
				if (delete){
					treeptr = bestTree;
					numtotalPath--;
					dprevouterbestlog[numtotalPath-1] = currentbestlike;
				}
			} else {
				break;
			}
			/****************************************************/
			
			/****************************************************/
			if(numtotalPath > 1){
				if(dprevouterbestlog[numtotalPath-1] < dprevouterbestlog[numtotalPath-2]) {
					peaklike = dprevouterbestlog[numtotalPath-2];
				} else if((dprevouterbestlog[numtotalPath-1]*EPSILON > dprevouterbestlog[numtotalPath-2]) && dprevouterbestlog[numtotalPath-1] > peaklike){
					finalTree = (Casenode) bestTree.clone();
				} else if(numtotalPath <= nminchild) {
					finalTree = (Casenode) bestTree.clone();
				}
			}
			/****************************************************/
			
			bendsearchlocal = Main_interface.bendsearch; //bend searching
			casetext.append(" Current path: "+numtotalPath+"; Value function: " + currentbestlike+"\n");
			casetext.paintImmediately(casetext.getBounds());
			
			for(int path=numtotalPath-1; path>Math.max(numtotalPath-6, 0); path--){
				if(dprevouterbestlog[path]*EPSILON > dprevouterbestlog[path-1]){
					keepon = true;
					break;
				}
			}
		} while(!bendsearchlocal && (((keepon || numtotalPath==1) && numtotalPath < nmaxchild) || (numtotalPath < nminchild)));
		treeptr = finalTree;
	}


	/**
	 * Builds the initial tree which is just a single chain with mean and
	 * standard deviation for each node being the global mean and standard
	 * deviation at the time point
	 */
	public void buildEmptyTree(Casenode currnode, double[] curve) {
			currnode.numchildren = 1;
			currnode.nextptr[0] = new Casenode(currnode);
			currnode.nextptr[0].curve = curve;
			currnode.nextptr[0].parent = currnode;
	}

	/**
	 * Deletes a child from the specified path on a cloned version of the root
	 * node
	**/
	public Casenode deletepath(int path, Casenode root) {
		Casenode treeroot = (Casenode) root.clone();
		Casenode ptr = treeroot;

		for (int nj = path + 1; nj < ptr.numchildren; nj++) {
			ptr.nextptr[nj - 1] = ptr.nextptr[nj];
		}
		ptr.numchildren--;
		return treeroot;
	}
	
	private void traverseandadd(Casenode origroot, Integer nchild, List<Integer> list1, List<Integer> list2) 
			throws Exception {
		if (!Main_interface.bendsearch && (origroot != null) && (origroot.numchildren < nmaxchild)) {
			//System.out.println(numtotalPath+" "+origroot.numchildren);
				Casenode splittree = splitnode(origroot, nchild, list1, list2); 
				if (splittree != null) {  //If succeed to add path
					double dlog = trainhmm(splittree, 100);	
					if (dlog > currentbestlike) {
						bestTree = splittree;
						currentbestlike = dlog;
					}
					if (dlog > dbestlog) {
						dbestlog = dlog;
					}
			    }
		}
	}

	private Casenode splitnode(Casenode origroot, Integer nchild, List<Integer> list1, List<Integer> list2) {
		
		Casenode treeroot = (Casenode) origroot.clone();
		Casenode ptr = treeroot;
		
		BuildSSM ssm1 = new BuildSSM();
		double[] curve1 = ssm1.meancurve(traindata, list1); //new path
		BuildSSM ssm2 = new BuildSSM();
		double[] curve2 = ssm2.meancurve(traindata, list2, ptr.nextptr[nchild].dcorrec); //old path
		
		ptr.nextptr[nchild].curve = curve2;
	    ptr.numchildren++;
		ptr.nextptr[ptr.numchildren - 1] = new Casenode(ptr);
		ptr.nextptr[ptr.numchildren - 1].curve = curve1;
		
		return treeroot;
	}


	/**
	 * Deletes the specificed path from the model starting from ndesiredlevel
	 */
	public void deleteMinPath(int path, Casenode root) {
		Casenode ptr = root;
		for (int nj = path + 1; nj < ptr.numchildren; nj++) {
			ptr.nextptr[nj - 1] = ptr.nextptr[nj];
		}
		ptr.numchildren--;
	}

	/**
	 * Helper function that searches for the best path to delete
	 */
	private boolean traverseanddelete(Casenode origroot) throws Exception {
		double bestdeletlog = dprevouterbestlog[numtotalPath-2];
		boolean delete = false;
		for (int nchild = 0; nchild < treeptr.numchildren; nchild++){
				Casenode deletetree = deletepath(nchild, origroot);
				if (deletetree != null) {
					double dlog = trainhmm(deletetree, 100);			
					if (dlog > bestdeletlog)
					{
						bestdeletlog = dlog;
						bestTree = deletetree;
						dbestlog = dlog;
						currentbestlike = dlog;
						delete = true;
					}
				}
			}
			return delete;
	}
	
	/**
	 * A Casenode corresponds to a state in the model
	 */
	public class Casenode {
		int ncurrtime;// use to count visited
		double[] curve;
		List<Integer> genelist;
		List<Integer> posi_genelist;
		List<Integer> nega_genelist; 
		double[] dcorrec;
		double[] dlikerec;
		int numchildren;
		
		List<String[][]> enrichment;
		Set<String> directnetwork;
		Set<String> coexpressionnetwork;
		HashMap<String, Set<String>> tftg;
		
		double[] dEsum;
		double[] dPsum;
		
		/** pointer to each of the next states */
		Casenode[] nextptr;
		/** pointer back to the parent */
		Casenode parent;
		
		/**
		 * Calls inittreenode with a null parent
		 */
		Casenode() {
			inittreenode(null);
		}

		Casenode(Casenode parent) {
			inittreenode(parent);
		}

		/**
		 * Calls inittreenode with parent
		 */
		void inittreenode(Casenode parent) {
			curve = new double[numcols];
			genelist = new ArrayList<Integer>();
			posi_genelist = new ArrayList<Integer>();
			nega_genelist = new ArrayList<Integer>();
			enrichment = new ArrayList<String[][]>();
			dEsum = new double[numcols];
			dPsum = new double[numcols];
			dcorrec = new double[numrows1];
			dlikerec = new double[numrows1];
			this.parent = parent;
			nextptr = new Casenode[nmaxchild];
			numchildren = 0;
			for (int nindex = 0; nindex < nextptr.length; nindex++) {
				nextptr[nindex] = null;
			}
		}

		/**
		 * For making copy of nodes
		 */
		public Object clone() {
		    Casenode tnode = new Casenode();
			tnode.curve = new double[curve.length];
			for(int j=0;j<tnode.curve.length;j++) tnode.curve[j] = curve[j];
			tnode.dcorrec = new double[dcorrec.length];
			for(int j=0;j<tnode.dcorrec.length;j++) tnode.dcorrec[j] = dcorrec[j];
			tnode.dlikerec = new double[dlikerec.length];
			for(int j=0;j<tnode.dlikerec.length;j++) tnode.dlikerec[j] = dlikerec[j];
			tnode.numchildren = numchildren;
			tnode.parent = null;
			tnode.nextptr = new Casenode[nmaxchild]; //瀹氫箟褰撳墠鑺傜偣鐨勫瓙鑺傜偣缁�
			for (int nindex = 0; nindex < nextptr.length; nindex++) {
				if (nextptr[nindex] == null) {
					tnode.nextptr[nindex] = null;
				} else {
					// cloning a new node
					tnode.nextptr[nindex] = (Casenode) nextptr[nindex].clone(); //null
					tnode.nextptr[nindex].parent = tnode;
				}
			}
			return tnode;  //杩斿洖褰撳墠鑺傜偣tnode
		}
	}
	
	/**
	 * Clears any prior assignments of genes to paths through the model
	 */
	public void clearCounts(Casenode ptr){
		if (ptr != null) {
			ptr.genelist.clear();
			ptr.posi_genelist.clear();
			ptr.nega_genelist.clear();
			for (int nchild = 0; nchild < ptr.numchildren; nchild++) {
				clearCounts(ptr.nextptr[nchild]);
			}
		}
	}
	
	public double hardclustering(Casenode treeptr){
		clearCounts(treeptr);
		double avgcorr = 0;
		for (int nrow = 0; nrow < numrows1; nrow++) {
			double bestcorr = 0;
			int bestchild = -1;
			for (int nchild = 0; nchild < treeptr.numchildren; nchild++){
				double corr = treeptr.nextptr[nchild].dcorrec[nrow];
				if(Math.abs(corr) > Math.abs(bestcorr)){
					bestcorr = corr;
					bestchild = nchild;
				}
			}
			avgcorr += Math.abs(bestcorr);
			if(bestcorr > 0){
				treeptr.nextptr[bestchild].genelist.add(nrow);
				treeptr.nextptr[bestchild].posi_genelist.add(nrow);
			}else if(bestcorr < 0){
				treeptr.nextptr[bestchild].genelist.add(nrow);
				treeptr.nextptr[bestchild].nega_genelist.add(nrow);
			}
		}
		avgcorr = avgcorr/numrows1;
		return avgcorr;
	}
	
	public void softclustering(Casenode treeptr){
		casetext.append("Assign seed genes based on soft clustering..."+"\n");
		casetext.paintImmediately(casetext.getBounds());
		casetext.append("================================"+"\n");
		casetext.paintImmediately(casetext.getBounds());
		clearCounts(treeptr);
		for (int nrow = 0; nrow < numrows1; nrow++) {
			double bestcorr = 0;
			int bestchild = -1;
			for (int nchild = 0; nchild < treeptr.numchildren; nchild++){
				double corr = treeptr.nextptr[nchild].dcorrec[nrow];
				if(Math.abs(corr) > Math.abs(bestcorr)){
					bestcorr = corr;
					bestchild = nchild;
				}
			}
			if(bestcorr > 0){
				treeptr.nextptr[bestchild].genelist.add(nrow);
				treeptr.nextptr[bestchild].posi_genelist.add(nrow);
			}else if(bestcorr < 0){
				treeptr.nextptr[bestchild].genelist.add(nrow);
				treeptr.nextptr[bestchild].nega_genelist.add(nrow);
			}
		}
		for (int nrow = 0; nrow < numrows2; nrow++) {
			int key = PPIgene1[nrow];
			for(int re = 0; re < PPIgene2[nrow].length; re++) {
				Integer[] genes = PPIgene2[nrow][re];
				Double[] values = PPIvalue[nrow][re];
				if(genes.length > 1) {
					double bestlike = 0;
					int bestchild = -1;
					for (int nchild = 0; nchild < treeptr.numchildren; nchild++){
						double like = 0;
						for(int i=0;i<genes.length;i++) {
							like += values[i]*treeptr.nextptr[nchild].dlikerec[genes[i]];
						}
						if(like > bestlike) {
							bestlike = like;
							bestchild = nchild;
						}
					}
					if(bestchild > 0) {
						if(!treeptr.nextptr[bestchild].genelist.contains(key)){
							treeptr.nextptr[bestchild].genelist.add(key);
							double bestcorr = treeptr.nextptr[bestchild].dcorrec[key];
							if(bestcorr > 0){
								treeptr.nextptr[bestchild].posi_genelist.add(key);
							} else if(bestcorr < 0) {
								treeptr.nextptr[bestchild].nega_genelist.add(key);
							}
						}
					}
				}
			}
		}
	}
	
	public void softclustering2(Casenode treeptr){
		casetext.append("Assign functional units to modules..."+"\n");
		casetext.paintImmediately(casetext.getBounds());
		casetext.append("================================"+"\n");
		casetext.paintImmediately(casetext.getBounds());
		clearCounts(treeptr);	
		for (int nrow = 0; nrow < numrows1; nrow++) {
			boolean found = false;
			for (int nchild = 0; nchild < treeptr.numchildren; nchild++){
				double like = treeptr.nextptr[nchild].dlikerec[nrow];
				if(like > 0.95) {
					found = true;
					double corr = treeptr.nextptr[nchild].dcorrec[nrow];
					if(corr > 0){
						treeptr.nextptr[nchild].genelist.add(nrow);
						treeptr.nextptr[nchild].posi_genelist.add(nrow);
					} else if(corr < 0){
						treeptr.nextptr[nchild].genelist.add(nrow);
						treeptr.nextptr[nchild].nega_genelist.add(nrow);
					}
				}
			}
			if(!found) {
				double bestcorr = 0;
				int bestchild = -1;
				for (int nchild = 0; nchild < treeptr.numchildren; nchild++){
					double corr = treeptr.nextptr[nchild].dcorrec[nrow];
					if(Math.abs(corr) > Math.abs(bestcorr)){
						bestcorr = corr;
						bestchild = nchild;
					}
				}
				if(bestcorr > 0){
					treeptr.nextptr[bestchild].genelist.add(nrow);
					treeptr.nextptr[bestchild].posi_genelist.add(nrow);
				}else if(bestcorr < 0){
					treeptr.nextptr[bestchild].genelist.add(nrow);
					treeptr.nextptr[bestchild].nega_genelist.add(nrow);
				}
			}
		}
		
		for (int nrow = 0; nrow < numrows2; nrow++) {
			int key = PPIgene1[nrow];
			for(int re = 0; re < PPIgene2[nrow].length; re++) {
				boolean found = false;
				Integer[] genes = PPIgene2[nrow][re];
				Double[] values = PPIvalue[nrow][re];
				if(genes.length > 1) {
					//System.out.println(genes.length+"\t"+values.length);
					for (int nchild = 0; nchild < treeptr.numchildren; nchild++){
						double like = 0;
						for(int i=0;i<genes.length;i++) {
							like += values[i]*treeptr.nextptr[nchild].dlikerec[genes[i]];
						}
						if(like > 0.95) {
							found = true;
							if(!treeptr.nextptr[nchild].genelist.contains(key)){
								treeptr.nextptr[nchild].genelist.add(key);
								double corr = treeptr.nextptr[nchild].dcorrec[key];
								if(corr > 0){
									treeptr.nextptr[nchild].posi_genelist.add(key);
								} else if(corr < 0){
									treeptr.nextptr[nchild].nega_genelist.add(key);
								}
							}
						}
					}
					if(!found) {
						double bestlike = 0;
						int bestchild = -1;
						for (int nchild = 0; nchild < treeptr.numchildren; nchild++){
							double like = 0;
							for(int i=0;i<genes.length;i++) {
								like += values[i]*treeptr.nextptr[nchild].dlikerec[genes[i]];
							}
							if(like > bestlike) {
								bestlike = like;
								bestchild = nchild;
							}
						}
						if(bestchild > 0) {
							if(!treeptr.nextptr[bestchild].genelist.contains(key)){
								treeptr.nextptr[bestchild].genelist.add(key);
								double bestcorr = treeptr.nextptr[bestchild].dcorrec[key];
								if(bestcorr > 0){
									treeptr.nextptr[bestchild].posi_genelist.add(key);
								} else if(bestcorr < 0) {
									treeptr.nextptr[bestchild].nega_genelist.add(key);
								}
							}
						}
					}
				}
			}
		}
		//for(int i=0;i<treeptr.numchildren;i++) {
		//	System.out.println(treeptr.nextptr[i].posi_genelist.size()+"\t"+treeptr.nextptr[i].nega_genelist.size());
		//}
	}

	public void initAE(Casenode root) {
		if (root != null) {
			for (int nchild = 0; nchild < root.numchildren; nchild++) {
				initAE(root.nextptr[nchild]);
			}
			root.dcorrec = new double[numrows1];
			root.dlikerec = new double[numrows1];
			root.dEsum = new double[numcols];
			root.dPsum = new double[numcols];
		}
	}
	
	public void initND(Casenode root) {
		Sumcorr = 0;
		Sumcorrsq = 0;
		int NDcount = numrows1 * root.numchildren;
		for (int nrow = 0; nrow < numrows1; nrow++){
			 getcorr(root, nrow);
		}
		NDmean = Sumcorr / NDcount;
		double dval = (double)NDcount / (double)(NDcount - 1)
				* (Sumcorrsq / NDcount - Math.pow(NDmean, 2));
		NDsigma = Math.sqrt(dval);
		Normalcorr = new NormalDistribution(NDmean, NDsigma);
		//System.out.println(Normalcorr.getMean()+" "+Normalcorr.getStandardDeviation());
	}
	
	public void getcorr(Casenode node, int nrow) {
		Pearsonr pr = new Pearsonr();
		for (int nchild = 0; nchild < node.numchildren; nchild++){
			if(node.nextptr[nchild].curve != null){
				double corr = pr.cosineSimilarity(traindata[nrow], node.nextptr[nchild].curve);
				Sumcorr += corr;
				Sumcorrsq += Math.pow(corr, 2);
				node.nextptr[nchild].dcorrec[nrow] = corr;
			}
		}
	}
	
	public void instanceAE(Casenode root, int bestchild, double[] vals, int nrec) throws Exception {
			double corr = root.nextptr[bestchild].dcorrec[nrec];
			if(corr>0){
				for(int i=0;i<vals.length;i++){
					root.nextptr[bestchild].dEsum[i] += vals[i];
					root.nextptr[bestchild].dPsum[i] += 1;
				}
			} else {
				for(int i=0;i<vals.length;i++){
					root.nextptr[bestchild].dEsum[i] -= vals[i];
					root.nextptr[bestchild].dPsum[i] += 1;
				}
		  }
	}
	
	public double instanceAE(Casenode root, int nrec) throws Exception {
		double dlikerec = 0;
		for(int i=0;i<PPIgene2[nrec].length;i++) {
			Integer[] genes = PPIgene2[nrec][i];
			Double[] weights = PPIvalue[nrec][i];
			if(genes.length > 1) {
				double bestlike = 0;
				int bestchild = -1;
				for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
					double like = 0;
					for(int j=0;j<genes.length;j++) {
						like += weights[j]*treeptr.nextptr[nchild].dlikerec[genes[j]];
					}
					if(like > bestlike) {
						bestlike = like;
						bestchild = nchild;
					}
				}
				
				if(bestlike > 0) {
					dlikerec += bestlike;
					for(int j=0;j<genes.length;j++) {
						double corr = treeptr.nextptr[bestchild].dcorrec[genes[j]];
						if(corr > 0){
							for(int m=0;m<traindata[0].length;m++){
								root.nextptr[bestchild].dEsum[m] += weights[j]*traindata[genes[j]][m];
								root.nextptr[bestchild].dPsum[m] += weights[j];
							}
						} else {
							for(int m=0;m<traindata[0].length;m++){
								root.nextptr[bestchild].dEsum[m] -= weights[j]*traindata[genes[j]][m];
								root.nextptr[bestchild].dPsum[m] += weights[j];
							}
					    }
					}
				}
			}
		}
		return dlikerec;
    }
	
	public void updateParams(Casenode root) throws Exception {
		for (int nchild = 0; nchild < root.numchildren; nchild++) {
			Casenode ptr = root.nextptr[nchild];
			for(int i=0;i<numcols;i++){
				ptr.curve[i] = ptr.dEsum[i]/ptr.dPsum[i];
			}
		}
	}
	
	
	public double trainhmm(Casenode treehmm, int breaker)throws Exception {
		double dlike = 0;
		double dbestlike = 0;
		initAE(treehmm);
		initND(treehmm);
		
		Gettfbacklg backlg = new Gettfbacklg();
		backlg.multithread(treehmm, Normalcorr, minPearson, numrows1, 400);
		int[] bestchildrec = backlg.bestchildrec;
		double[] bestlikerec = backlg.bestlikerec;
		for (int nrow = 0; nrow < numrows1; nrow++){
			if(bestlikerec[nrow] > 0){
				instanceAE(treehmm, bestchildrec[nrow], traindata[nrow], nrow);
				dlike += bestlikerec[nrow];
			}
		}
		for (int nrow = 0; nrow < numrows2; nrow++){
			dlike += instanceAE(treehmm, nrow);			
		}

		dbestlike = dlike;
		updateParams(treehmm);
		
		dlike = 0;
		initAE(treehmm);
		initND(treehmm);
		backlg.multithread(treehmm, Normalcorr, minPearson, numrows1, 400);
		bestchildrec = backlg.bestchildrec;
		bestlikerec = backlg.bestlikerec;
		for (int nrow = 0; nrow < numrows1; nrow++){
			if(bestlikerec[nrow] > 0){
				instanceAE(treehmm, bestchildrec[nrow], traindata[nrow], nrow);
				dlike += bestlikerec[nrow];
			}
		}
		for (int nrow = 0; nrow < numrows2; nrow++){
			dlike += instanceAE(treehmm, nrow);
		}
		
		int count = 0;
		while((Math.abs(dlike - dbestlike) / dlike) > BEPSILON){
			dbestlike = dlike;
			updateParams(treehmm);
			dlike = 0;
			initAE(treehmm);
			initND(treehmm);
			backlg.multithread(treehmm, Normalcorr, minPearson, numrows1, 400);
			bestchildrec = backlg.bestchildrec;
			bestlikerec = backlg.bestlikerec;
			for (int nrow = 0; nrow < numrows1; nrow++){
				if(bestlikerec[nrow] > 0){
					instanceAE(treehmm, bestchildrec[nrow], traindata[nrow], nrow);
					dlike += bestlikerec[nrow];
				}
			}
			for (int nrow = 0; nrow < numrows2; nrow++){
				dlike += instanceAE(treehmm, nrow);
			}
			count++;
			if(count > breaker){
				break;
			}
		}
		System.out.println(count+" "+dlike);
		return (dlike);
	}
	
	public void trainhmm2(Casenode treehmm)throws Exception {
		initAE(treehmm);
		initND(treehmm);
		Gettfbacklg backlg = new Gettfbacklg();
		backlg.multithread(treehmm, Normalcorr, minPearson, numrows1, 400);
	}
}
