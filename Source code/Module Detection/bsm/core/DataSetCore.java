package bsm.core;

import java.io.*;
import java.util.*;
import java.util.zip.*;

/**
 *The class encapsulates a set of gene expression data
 */
public class DataSetCore {

	/**
	 *The designated main data file associated with the set, others are repeats
	 */
	public String szInputFile;

	/**
	 *If repeat data comes from distinct time series (true) that is a
	 * longitudal time series, otherwise it is false and each column of the same
	 * time point between data sets is interchangeable
	 */
	public boolean bfullrepeat;

	/**
	 *The names of the other repeat files
	 */
	public String[] otherInputFiles;

	/**
	 *Contains genes filtered. Maps these gene names to the list of probe IDs
	 * associated with it
	 */
	public HashMap<String,String> htFiltered;

	/**
	 *The maximum number of missing values to prevent a gene from being
	 * filtered.
	 */
	public int nmaxmissing;

	/**
	 *Minimum average pairwise correlation a gene must have between full
	 * repeats if bfullrepeat is true
	 */
	
	public int missinter;
	public boolean btakelog;
	
	public double dmincorrelation;

	/**
	 *Number of rows in the data matrix. This corresponds to the number of
	 * genes.
	 */
	public int numrows;

	/**
	 *Number of columns in the data matrix. This corresponds to the number of
	 * time points
	 */
	public int numcols;

	/**
	 *The expression data, row are genes, columns are time points in
	 * experiments
	 */ 
	public double[][] controldata;
	public double[][] controlnorm;
	/**
	 *0 if data value is missing non-zero if present
	 */
	public int[][] pmacontrol;

	/**
	 *True if the spot column was included in the data file
	 */
	public boolean bspotincluded;

	/**
	 *Present/missing data for one data set First dimension is gene Second
	 * dimension is spot Third is present/missing value
	 */
	public int[][][] genespotcontrolpma = null;
	public int[][][][] generepeatspotcontrolpma = null;
	public double[][][] genespotcontroldata = null;
	public double[][][][] generepeatspotcontroldata = null;
	
	/**
	 *The list of probe IDs in the current data set
	 */
	public String[] probenames;

	/**
	 *The list of gene names for the current data set
	 */
	public String[] genenames;

	public HashMap<String, Integer> gene2int;
	/**
	 *The distribution of all the average pairwise correlations of genes across
	 * full repeats
	 */
	public double[] sortedcorrvals = null;

	/**
	 *The time points at which the expression data was sampled
	 */
	public String[] dsamplemins;

	/**
	 *The threshold value for required change
	 */
	public double dthresholdvalue = -1;

	/**
	 * True if gene change threshold for filtering is based on the max-min
	 * difference False if the gene change threshold for filtering is based on
	 * the absolute difference
	 */
	public boolean bmaxminval;
	public boolean bfcto0;
	public boolean bfctopre;

	/**
	 * True if a column of inital 0's should be added to the data file, false
	 * otherwise
	 */
	public boolean badd0 = false;

	/**
	 *The header string for the spot ID column
	 */
	public String szProbeHeader;

	/**
	 *The header string for the gene name column
	 */
	public String szGeneHeader;

	/**
	 * Empty constructor
	 */
	public DataSetCore() {

	}

	/**
	 *Constructor for individual variables
	 */
	private DataSetCore(String szInputFile, double[][][] genespotcontroldata,
			double[][][][] generepeatspotcontroldata, int[][][] genespotcontrolpma,
			int[][][][] generepeatspotcontrolpma, double[][] controldata, double[][] controlnorm,
			int[][] pmacontrol, String[] probenames, String[] genenames, 
			HashMap<String, Integer> gene2int, double dthresholdvalue, boolean btakelog, 
			boolean bspotincluded, boolean badd0, String[] dsamplemins, double[] sortedcorrvals,
			int nmaxmissing, int missinter, double dmincorrelation, HashMap<String,String> htFiltered, 
			String szProbeHeader, String szGeneHeader, boolean bmaxminval, boolean bfcto0, boolean bfctopre,
			String[] otherInputFiles, boolean bfullrepeat) {
		this.otherInputFiles = otherInputFiles;
		this.bfullrepeat = bfullrepeat;
		this.szInputFile = szInputFile;
		this.bmaxminval = bmaxminval;
		this.bfcto0 = bfcto0;
		this.bfctopre = bfctopre;
		this.htFiltered = htFiltered;
		this.szProbeHeader = szProbeHeader;
		this.szGeneHeader = szGeneHeader;
		this.dmincorrelation = dmincorrelation;
		this.nmaxmissing = nmaxmissing;
		this.missinter = missinter;
		this.genespotcontroldata = genespotcontroldata;
		this.generepeatspotcontroldata = generepeatspotcontroldata;
		this.genespotcontrolpma = genespotcontrolpma;
		this.generepeatspotcontrolpma = generepeatspotcontrolpma;
		this.controldata = controldata;
		this.controlnorm = controlnorm;
		this.pmacontrol = pmacontrol;
		this.probenames = probenames;
		this.genenames = genenames;
		this.gene2int = gene2int;
		this.dthresholdvalue = dthresholdvalue;
		this.btakelog = btakelog;
		this.bspotincluded = bspotincluded;
		this.badd0 = badd0;
		numrows = controldata.length;
		numcols = controldata[0].length;;
		this.dsamplemins = dsamplemins;
		this.sortedcorrvals = sortedcorrvals;
	}

	// //////////////////////////////////////////////////////////////////
	/**
	 *Constructor copies each field
	 */
	public DataSetCore(DataSetCore theDataSetCore) {
		this.szInputFile = theDataSetCore.szInputFile;
		this.bfullrepeat = theDataSetCore.bfullrepeat;
		this.otherInputFiles = theDataSetCore.otherInputFiles;
		this.htFiltered = theDataSetCore.htFiltered;
		this.nmaxmissing = theDataSetCore.nmaxmissing;
		this.missinter = theDataSetCore.missinter;
		this.bmaxminval = theDataSetCore.bmaxminval;
		this.bfcto0 = theDataSetCore.bfcto0;
		this.bfctopre = theDataSetCore.bfctopre;
		this.dmincorrelation = theDataSetCore.dmincorrelation;
		this.numrows = theDataSetCore.numrows;
		this.numcols = theDataSetCore.numcols;
		this.controldata = theDataSetCore.controldata;
		this.controlnorm = theDataSetCore.controlnorm;
		this.bspotincluded = theDataSetCore.bspotincluded;
		this.genespotcontrolpma = theDataSetCore.genespotcontrolpma;
		this.genespotcontroldata = theDataSetCore.genespotcontroldata;
		this.generepeatspotcontrolpma = theDataSetCore.generepeatspotcontrolpma;
		this.generepeatspotcontroldata = theDataSetCore.generepeatspotcontroldata;
		this.pmacontrol = theDataSetCore.pmacontrol;
		this.btakelog = theDataSetCore.btakelog;
		this.dthresholdvalue = theDataSetCore.dthresholdvalue;
		this.probenames = theDataSetCore.probenames;
		this.genenames = theDataSetCore.genenames;
		this.gene2int = theDataSetCore.gene2int;
		this.sortedcorrvals = theDataSetCore.sortedcorrvals;
		this.dsamplemins = theDataSetCore.dsamplemins;
		this.badd0 = theDataSetCore.badd0;
		this.szProbeHeader = theDataSetCore.szProbeHeader;
		this.szGeneHeader = theDataSetCore.szGeneHeader;
	}



	// /////////////////////////////////////////////////////////////////////////////////////////
	/**
	 *读存在szInputFile中的文件名对应的文件
	 */
	protected void dataSetReader(String szInputFile, int missinter, boolean btakelog)
			throws IOException, FileNotFoundException, IllegalArgumentException {
		htFiltered = new HashMap<String,String>();
		if (szInputFile.equals("")) {
			throw new IllegalArgumentException("No input file specified!");
		}

		BufferedReader brInputFile;
		// first tries reading GZIPInputStream format
		try {
			brInputFile = new BufferedReader(new InputStreamReader(
					new GZIPInputStream(new FileInputStream(szInputFile))));
		} catch (IOException ex) {
			brInputFile = new BufferedReader(new FileReader(szInputFile));
		}

		String szLine, szToken;
		StringTokenizer st, st2;
		String szHeaderLine;
		
		szHeaderLine = brInputFile.readLine();
		if (szHeaderLine == null) {
			throw new IllegalArgumentException("Input File " + szInputFile + " is empty!");
		} else {
			while (szHeaderLine.equals("")) {
				szHeaderLine = brInputFile.readLine();
				if (szHeaderLine == null) {
					throw new IllegalArgumentException("Input File "+ szInputFile + " is empty!");
				}
			}
		}

		// checks if every col ends with a tab then we assume we have one less
		// column
		boolean balltabend = true;
		// stores the expression data into alInputFile
		ArrayList<String> alInputFile = new ArrayList<String>();
		while ((szLine = brInputFile.readLine()) != null) {
			StringTokenizer szblank = new StringTokenizer(szLine, " \t");

			if ((szblank.countTokens() >= 1) || (!bspotincluded)) {
				alInputFile.add(szLine);
				if (balltabend) {
					balltabend = (szLine.endsWith("\t"));
				}
			}
		}
		brInputFile.close();

		

		// stores the expression data into alInputFile
		
		st = new StringTokenizer(szHeaderLine, "\t");
		st2 = new StringTokenizer(szHeaderLine, "\t", true);

		numcols = st2.countTokens() - st.countTokens();
		if (balltabend) {
			numcols--;
		}

		if (bspotincluded) {
			numcols--;
		}

		if (badd0) {
			numcols++;
		}

		dsamplemins = new String[Math.max(numcols, 1)];

		if (bspotincluded) {
			szProbeHeader = st2.nextToken();
			if (!szProbeHeader.equals("\t")) {
				if (!st2.hasMoreTokens()) {
					String szmsg = "Missing gene header.";
					if (numcols == -1) {
						szmsg += "\nConsider unchecking 'Spot IDs included in the data file'";
					}
					throw new IllegalArgumentException(szmsg);
				}
				st2.nextToken();
			} else {
				szProbeHeader = "";
			}
		} else {
			szProbeHeader = "SPOT";
		}

		szGeneHeader = st2.nextToken(); // gene header
		if (!szGeneHeader.equals("\t")) {
			if (st2.hasMoreTokens()) {
				st2.nextToken(); // flush tab
			}
		} else {
			szGeneHeader = ""; // no gene header given
		}

		int npoint;
		if (badd0) {
			dsamplemins[0] = "0";
			npoint = 1;
		} else {
			npoint = 0;
		}

		for (; npoint < numcols; npoint++) {
			if (st2.hasMoreTokens()) {
				szToken = st2.nextToken();
			} else {
				throw new IllegalArgumentException("Missing a column header");
			}

			if (!szToken.equals("\t")) {
				if (st2.hasMoreTokens()) {
					st2.nextToken();
				}
			} else {
				szToken = "";
			}
			dsamplemins[npoint] = szToken;
		}
		
		
		numrows = alInputFile.size();
		if (numrows == 0) {
			throw new IllegalArgumentException(szInputFile + " is empty!");
		}
		
		numcols = dsamplemins.length;
		controldata = new double[numrows][numcols];
		pmacontrol = new int[numrows][numcols];
		probenames = new String[numrows];
		genenames = new String[numrows];
		gene2int = new HashMap<String, Integer>();
		String sztoken;
		HashMap htProbeIDs = new HashMap();
		double DLOG2 = Math.log(2);

		for (int nrow = 0; nrow < numrows; nrow++) {
			szLine = (String) alInputFile.get(nrow);

			st = new StringTokenizer(szLine, "\t", true);
			if (bspotincluded) {
				sztoken = st.nextToken();
				// blank probes not allowed, nor are duplicates
				if (sztoken.equals("\t")) {
					String szmsg;
					szmsg = "Missing a Spot Name";

					throw new IllegalArgumentException(szmsg);
				} else {
					if (htProbeIDs.containsKey(sztoken)) {
						String szmsg;
	
						szmsg = "Spot name " + sztoken
									+ " in repeat/comparison is not unique";

						throw new IllegalArgumentException(szmsg);
					} else {
						htProbeIDs.put(sztoken, null);
						probenames[nrow] = sztoken.trim().toUpperCase(
								Locale.ENGLISH);
					}

					if (st.hasMoreTokens()) {
						st.nextToken(); // flush token
					}
				}
			} else {
				probenames[nrow] = "ID_" + nrow;
			}

			if (!st.hasMoreTokens()) {
				genenames[nrow] = null;
				for (int ncol = 0; ncol < numcols; ncol++) {
					pmacontrol[nrow][ncol] = 0;
				}
			} else {
				sztoken = st.nextToken();
				if ((sztoken.equals("\t")) || (sztoken.equals("0"))) {
					// gene name missing; 0 counts as missing name field
					if ((sztoken.equals("0")) && (st.hasMoreTokens())) {
						st.nextToken();
					}
					genenames[nrow] = null;
				} else {
					if ((sztoken.charAt(0) == '\"')
							&& (sztoken.charAt(sztoken.length() - 1) == '\"')) {
						// strings quotes
						sztoken = sztoken.substring(1, sztoken.length() - 1);
					}
					if (st.hasMoreTokens()) {
						st.nextToken(); // get rid of tab
					}
					genenames[nrow] = sztoken.trim().toUpperCase(Locale.ENGLISH);
					gene2int.put(genenames[nrow], nrow);
				}

				boolean beol = false;
				double sumcontrol = 0;
				int countcontrol = 0;
				int indexcontrol = 0;
				double mincontrol = Double.POSITIVE_INFINITY;

				
				for (int ncol = 0; ncol < numcols; ncol++) {
					if (!st.hasMoreTokens()) {
						beol = true;
					} else {
						sztoken = st.nextToken();
					}

					
					if ((beol) || (sztoken.equals("\t"))) {
						pmacontrol[nrow][indexcontrol] = 0;
						indexcontrol++;
					} else {
						try {						
								if (btakelog) {
									if(Double.parseDouble(sztoken) <= 0){
										throw new IllegalArgumentException("Control value <= 0 at (row:"+(nrow+1)+" column:"
												+(indexcontrol+1)+") for log normalization.");
									}else{
										controldata[nrow][indexcontrol] = Math.log(Double.parseDouble(sztoken))/DLOG2;
									}
								}else{
									controldata[nrow][indexcontrol] = Double.parseDouble(sztoken);
								}
								pmacontrol[nrow][indexcontrol] = 2;
								sumcontrol += controldata[nrow][indexcontrol];
								if(mincontrol>controldata[nrow][indexcontrol]) mincontrol = controldata[nrow][indexcontrol];
								countcontrol++;
								indexcontrol++;
						} catch (NumberFormatException pe) {
							String szmsg = sztoken + " is not a valid real number";
							throw new IllegalArgumentException(szmsg);
						}

						if (st.hasMoreTokens()) {
							sztoken = st.nextToken();
							if (sztoken.equals("\n")) {
								beol = true;
							}
						}
					}
				}
				for (int ncol = 0; ncol < numcols; ncol++) {
					if(pmacontrol[nrow][ncol] == 0){
						if(missinter==1){
							controldata[nrow][ncol] = sumcontrol/countcontrol;
						}else if(missinter==0){
							controldata[nrow][ncol] = mincontrol;
						}else if(missinter==2){
							controldata[nrow][ncol] = 0.0;
						}
					}
				}
			}
		}
	}

	// //////////////////////////////////////////////////////////////
	/**
	 *剔除缺失值过大的行
	 */
	public DataSetCore filterMissing() {
		boolean[] goodrow = new boolean[numrows];
		int ngoodrows = 0;

		for (int nrow = 0; nrow < numrows; nrow++) {
			int controlmissing = 0;
			boolean bgoodrow = true;
			for (int ncol = 0; ncol < numcols; ncol++) {
				if(pmacontrol[nrow][ncol] == 0) controlmissing++;
			}
			bgoodrow = (controlmissing <= nmaxmissing);
			if (bgoodrow) {
				goodrow[nrow] = true;
				ngoodrows++;
			} else {
				goodrow[nrow] = false;
			}
		}

		return filtergenesgeneral(goodrow, ngoodrows, true);
	}

	// ///////////////////////////////////////////////////////////////
	/**
	 *删除与其他重复的行
	 */
	public DataSetCore filterDuplicates() {
		boolean[] goodrow = new boolean[numrows];
		int ngoodrows = 0;

		Hashtable<String,ArrayList<Integer>> htgenenames = new Hashtable<String,ArrayList<Integer>>();
		// built hashtable of name to index
		for (int nrow = 0; nrow < numrows; nrow++) {
			ArrayList<Integer> indicies = htgenenames.get(genenames[nrow]);
			if (indicies == null) {
				goodrow[nrow] = true;
				// this id is new
				indicies = new ArrayList<Integer>();
				ngoodrows++;
				htgenenames.put(genenames[nrow], indicies);
			} else {
				goodrow[nrow] = false;
				// already seen this id
				indicies.add(new Integer(nrow));
			}
		}

		return filtergenesgeneral(goodrow, ngoodrows, false);
	}

	public void normalization() {
		controlnorm = new double[numrows][numcols];
		for(int i=0;i<numrows;i++){
			double[] controlvalue = controldata[i];
			double controlmean = Util.getmean(controlvalue);
			double controlstd = Util.getstd(controlvalue);
			for(int j=0;j<controlvalue.length;j++){
				controlnorm[i][j] = (controlvalue[j] - controlmean)/controlstd;
			}
		}
	}
	
	
	
	public DataSetCore averageAndFilterDuplicates() {

		boolean[] goodrow = new boolean[numrows];
		int ngoodrows = 0;

		Hashtable<String,ArrayList<Integer>> htgenenames = new Hashtable<String,ArrayList<Integer>>();
		int nmaxsize = 1;
		// built hashtable of name to index
		for (int nrow = 0; nrow < numrows; nrow++) {
			if(genenames[nrow]!=null){
				goodrow[nrow] = true;
				ArrayList<Integer> indicies = htgenenames.get(genenames[nrow]);
				if (indicies == null) {
					// this id is new
					indicies = new ArrayList<Integer>();
					ngoodrows++;
					htgenenames.put(genenames[nrow], indicies);
				} else {
					// already seen this id
					indicies.add(new Integer(nrow));
					if (indicies.size() > nmaxsize) {
						nmaxsize = indicies.size();
					}
				}
			}
		}

		// going to store the values here then sort
		double[] vals = new double[nmaxsize + 1];

		int ngoodindex = -1;
		// store ratio and expression and pma values before merging genes
		genespotcontroldata = new double[ngoodrows][][];
		genespotcontrolpma = new int[ngoodrows][][];

		for (int nrow = 0; nrow < numrows; nrow++) {
			if (goodrow[nrow]) {
				ngoodindex++;
				ArrayList<Integer> indicies = htgenenames.get(genenames[nrow]);
				int nindiciessize = indicies.size();

				genespotcontroldata[ngoodindex] = new double[nindiciessize + 1][];
				genespotcontroldata[ngoodindex][0] = controldata[nrow];
				genespotcontrolpma[ngoodindex] = new int[nindiciessize + 1][];
				genespotcontrolpma[ngoodindex][0] = pmacontrol[nrow];
				
				// other rows may match this one
				for (int ncol = 0; ncol < numcols; ncol++) {
					// go through each column finding median
					int nvalindex = 0;
					int npma = 0;
					if (pmacontrol[nrow][ncol] != 0) {
						vals[nvalindex] = controldata[nrow][ncol];
						nvalindex++;
						npma = pmacontrol[nrow][ncol];
					}

					for (int nindex2 = 0; nindex2 < nindiciessize; nindex2++) {
						// go through all matches to this one
						int nrow2 = ((Integer) indicies.get(nindex2)).intValue();

						if (ncol == 0) {
							genespotcontroldata[ngoodindex][nindex2 + 1] = controldata[nrow2];
							genespotcontrolpma[ngoodindex][nindex2 + 1] = pmacontrol[nrow2];
							goodrow[nrow2] = false;
							probenames[nrow] += ";" + probenames[nrow2];
						}

						if (pmacontrol[nrow2][ncol] != 0) {
							// this is a valid data value
							vals[nvalindex] = controldata[nrow2][ncol];
							nvalindex++;
							npma = Math.max(npma, pmacontrol[nrow2][ncol]);
						}
					}

					pmacontrol[nrow][ncol] = npma;
					if (nvalindex > 0) {
						// averages using median
						controldata[nrow][ncol] = Util.getmean(vals, nvalindex);
					}
				}
			}
		}
		return filtergenesgeneral(goodrow, ngoodrows, false);
	}
	
	/**
	 *删除重复行并取最大值
	 */
	public DataSetCore maxAndFilterDuplicates() {

		boolean[] goodrow = new boolean[numrows];
		int ngoodrows = 0;

		Hashtable<String,ArrayList<Integer>> htgenenames = new Hashtable<String,ArrayList<Integer>>();
		int nmaxsize = 1;
		// built hashtable of name to index
		for (int nrow = 0; nrow < numrows; nrow++) {
			if(genenames[nrow]!=null){
				goodrow[nrow] = true;
				ArrayList<Integer> indicies = htgenenames.get(genenames[nrow]);
				if (indicies == null) {
					// this id is new
					indicies = new ArrayList<Integer>();
					ngoodrows++;
					htgenenames.put(genenames[nrow], indicies);
				} else {
					// already seen this id
					indicies.add(new Integer(nrow));
					if (indicies.size() > nmaxsize) {
						nmaxsize = indicies.size();
					}
				}
			}
		}

		// going to store the values here then sort
		double[] vals = new double[nmaxsize + 1];

		int ngoodindex = -1;
		// store ratio and expression and pma values before merging genes
		genespotcontroldata = new double[ngoodrows][][];
		genespotcontrolpma = new int[ngoodrows][][];

		for (int nrow = 0; nrow < numrows; nrow++) {
			if (goodrow[nrow]) {
				ngoodindex++;
				ArrayList<Integer> indicies = htgenenames.get(genenames[nrow]);
				int nindiciessize = indicies.size();

				genespotcontroldata[ngoodindex] = new double[nindiciessize + 1][];
				genespotcontroldata[ngoodindex][0] = controldata[nrow];
				genespotcontrolpma[ngoodindex] = new int[nindiciessize + 1][];
				genespotcontrolpma[ngoodindex][0] = pmacontrol[nrow];

				// other rows may match this one
				for (int ncol = 0; ncol < numcols; ncol++) {
					// go through each column finding median
					int nvalindex = 0;
					int npma = 0;
					if (pmacontrol[nrow][ncol] != 0) {
						vals[nvalindex] = controldata[nrow][ncol];
						nvalindex++;
						npma = pmacontrol[nrow][ncol];
					}

					for (int nindex2 = 0; nindex2 < nindiciessize; nindex2++) {
						// go through all matches to this one
						int nrow2 = ((Integer) indicies.get(nindex2)).intValue();

						if (ncol == 0) {
							genespotcontroldata[ngoodindex][nindex2 + 1] = controldata[nrow2];
							genespotcontrolpma[ngoodindex][nindex2 + 1] = pmacontrol[nrow2];
							goodrow[nrow2] = false;
							probenames[nrow] += ";" + probenames[nrow2];
						}

						if (pmacontrol[nrow2][ncol] != 0) {
							// this is a valid data value
							vals[nvalindex] = controldata[nrow2][ncol];
							nvalindex++;
							npma = Math.max(npma, pmacontrol[nrow2][ncol]);
						}
					}

					pmacontrol[nrow][ncol] = npma;
					if (nvalindex > 0) {
						// averages using median
						controldata[nrow][ncol] = Util.getmax(vals, nvalindex);
					}
				}
			}
		}
		return filtergenesgeneral(goodrow, ngoodrows, false);
	}
	
	// ///////////////////////////////////////////////////////////////////////////
	/**
	 *Converts data into log-ratio.
	 */
	public DataSetCore logratio2() {
		double DLOG2 = Math.log(2);
		if (btakelog) {
		for (int nrow = 0; nrow < numrows; nrow++) {
			/*
				for (int ncol = 0; ncol < numcols; ncol++) {
					data[nrow][ncol] = Math.log(data[nrow][ncol]) / DLOG2;
					if(ncol<controlcols){
						controldata[nrow][ncol] = data[nrow][ncol];
					}else{
						casedata[nrow][ncol-controlcols] = data[nrow][ncol];
					}
				}
				*/
				for (int ncol = 0; ncol < numcols; ncol++) {
					controldata[nrow][ncol] = Math.log(controldata[nrow][ncol]) / DLOG2;
				}
			}
		}
		return new DataSetCore(szInputFile, genespotcontroldata,
				generepeatspotcontroldata, genespotcontrolpma, generepeatspotcontrolpma,
				controldata, controlnorm, pmacontrol, probenames, genenames, gene2int, dthresholdvalue, 
				btakelog, bspotincluded, badd0, dsamplemins, sortedcorrvals, nmaxmissing, 
				missinter, dmincorrelation, htFiltered, szProbeHeader, szGeneHeader, bmaxminval, 
				bfcto0, bfctopre, otherInputFiles, bfullrepeat);
	}

	// //////////////////////////////////////////////////////////////////////////////
	/**
	 *Filters those rows which do not have a true in keepgene nkeep is the
	 * number of true rows in keepgene If bstore is true and gene is filtered
	 * then we stroe the gene and proble list for it in htFiltered Returns a new
	 * DataSetCore object with those rows filtered
	 */
	public DataSetCore filtergenesgeneral(boolean[] keepgene, int nkeep, boolean bstore) {
		if (nkeep < 1) throw new IllegalArgumentException("All Genes Filtered");

		double[][] filtercontrol = new double[nkeep][];
		int[][] filterpmacontrol = new int[nkeep][];
		double[][] filtercontrolnorm = new double[nkeep][];
		
		String[] filtergenenames = new String[nkeep];
		HashMap<String, Integer> filtergene2int = new HashMap<String, Integer>();
		String[] filterprobenames = new String[nkeep];

		double[][][] filtergenespotcontroldata;
		double[][][][] filtergenerepeatspotcontroldata;
		int[][][] filtergenespotcontrolpma;
		int[][][][] filtergenerepeatspotcontrolpma;

		boolean bfiltergenespottime = ((genespotcontroldata != null) && (genespotcontroldata.length == numrows));
		boolean bfiltergenerepeatspottime = ((generepeatspotcontroldata != null) && (generepeatspotcontroldata.length == numrows));

		if (!bfiltergenespottime) {
			filtergenespotcontroldata = genespotcontroldata;
			filtergenespotcontrolpma = genespotcontrolpma;
		} else {
			filtergenespotcontroldata = new double[nkeep][][];
			filtergenespotcontrolpma = new int[nkeep][][];
		}

		if (!bfiltergenerepeatspottime) {
			filtergenerepeatspotcontroldata = generepeatspotcontroldata;
			filtergenerepeatspotcontrolpma = generepeatspotcontrolpma;
		} else {
			filtergenerepeatspotcontroldata = new double[nkeep][][][];
			filtergenerepeatspotcontrolpma = new int[nkeep][][][];
		}

		int nfilterindex = 0;
		for (int nrow = 0; nrow < numrows; nrow++) {
			if (keepgene[nrow]) {
				// this row is a keeper store over its info
				filtergenenames[nfilterindex] = genenames[nrow];
				filtergene2int.put(filtergenenames[nfilterindex], nfilterindex);
				filterprobenames[nfilterindex] = probenames[nrow];
				
				if (bfiltergenespottime) {
					filtergenespotcontroldata[nfilterindex] = genespotcontroldata[nrow];
					filtergenespotcontrolpma[nfilterindex] = genespotcontrolpma[nrow];
				}

				if (bfiltergenerepeatspottime) {
					filtergenerepeatspotcontroldata[nfilterindex] = generepeatspotcontroldata[nrow];
					filtergenerepeatspotcontrolpma[nfilterindex] = generepeatspotcontrolpma[nrow];
				}

				filterpmacontrol[nfilterindex] = pmacontrol[nrow];
				filtercontrol[nfilterindex] = controldata[nrow];
				if(controlnorm != null) filtercontrolnorm[nfilterindex] = controlnorm[nrow];
				nfilterindex++;
			} else if (bstore) {
				htFiltered.put(genenames[nrow], probenames[nrow]);
			}
		}

		return new DataSetCore(szInputFile, filtergenespotcontroldata,
				filtergenerepeatspotcontroldata, filtergenespotcontrolpma,
				filtergenerepeatspotcontrolpma, filtercontrol, filtercontrolnorm,
				filterpmacontrol, filterprobenames, filtergenenames, filtergene2int,
				dthresholdvalue, btakelog, bspotincluded, badd0, dsamplemins,
				sortedcorrvals, nmaxmissing, missinter, dmincorrelation, htFiltered, 
				szProbeHeader, szGeneHeader, bmaxminval, bfcto0, bfctopre, otherInputFiles, bfullrepeat);
	}

}
