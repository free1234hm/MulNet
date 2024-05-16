package bsm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;
import org.apache.commons.math3.distribution.NormalDistribution;
import bsm.core.Correlationpro;
import bsm.core.Pearsonr;

public class RegulatorBindingData {
	/**
	 * The three dimensional matrix giving the temporal interaction values for
	 * genes and regulators. First dimension: time point Second dimension: gene
	 * Third dimension: regulator
	 */
	public double[][] gene2RegBinding;
	/**
	 * The signed version of gene2RegBinding. This gives the signed={-1,0,1}
	 * interaction value for the genes and regulators.
	 */
	public int[][] gene2RegBindingSigned;

	/**
	 * The signed version of reg2GeneBinding. This gives the signed={-1,0,1}
	 * interaction value for the genes and regulators.
	 */
	public int[][] reg2GeneBindingSigned;
	/**
	 * The three dimensional matrix gives the indices for the gene that a
	 * regulator binds with. First dimension: time point Second dimension:
	 * regulator Third dimension: gene
	 */
	public int[][] reg2GeneBindingIndex;
	/**
	 * The three dimensional matrix gives the indices for the regulator that a
	 * gene binds with. First dimension: time point Second dimension: gene Third
	 * dimension: regulator
	 */
	public int[][] gene2RegBindingIndex;
	
	public double[][] reg2GeneBinding;
	

	/**
	 * The number of Regulators aka the length of the regulatorNames array,
	 * previously numbits
	 */
	public int numberRegs;
	/**
	 * The number of genes in the data set.
	 */
	public int numberGenes;
	/**
	 * IDs of the regulators
	 */
	public String[] regNames;
	/**
	 * Synonyms of the names stored in regNames
	 */
	public HashMap<String, HashSet<String>> regSyns;
	/**
	 * Denotes the types of the regulators in the regulatorNames array i.e.,
	 * 0=TF and 1=miRNA for each entry in RegulatorNames
	 */
	public int[] regTypes;
	/**
	 * Maps the regulator names to their regulator type.
	 */
	public HashMap<String, Integer> regTypeMap;
	/**
	 * Maps the regulator names to their row in the regulator binding data.
	 */
	public HashMap<String, Integer> regRowMap;
	/**
	 * Can be queried for existing regulator types in regulatorTypes Each
	 * regulator type is an Integer TF = 0 miRNA = 1
	 */
	public HashSet<Integer> existingRegTypes;
	/**
	 * Contains scores associated with regulators computed by a RegulatorScorer
	 */
	public double[] regPriors;
	/**
	 * Unsorted possible binding values previously hsUniqueInput
	 */
	public HashSet<Double> existingBindingValuesUnsorted;
	/**
	 * Sorted possible binding values previously dbindingvals
	 */
	public double[] existingBindingValuesSorted;
	/**
	 * Sorted possible signed binding values For use by the filtered classified
	 */
	public int[] signedBindingValuesSorted;
	/**
	 * Unsorted possible signed binding values For use by the filtered
	 * classified
	 */
	public HashSet<Integer> signedBindingValuesUnsorted;
	/**
	 * Each element is a geneID associated with a BindingGeneRec object
	 * Effectively a duplicate of gene2RegulatorBinding previously htBinding
	 */
	public HashMap<String, BindingGeneRec> geneID2RegBinding;
	/**
	 * The gene to regulator data set up in a format to be used by a weighted
	 * classifier. The first dimension is the number of children, the second is
	 * gene number, and the third is regulator number, with the value being the
	 * binding value.
	 */
	/*
	 * Fields to use for filtering the data set if bfilterbinding == true
	 */
	public int nbinding;
	public boolean[] bbindingdata;

	/**
	 * All TF activiy priors are adjusted by ACTIVITY_EPSILON to avoid divide by
	 * 0 errors when the TF activity prior is 1.
	 */
	static final double ACTIVITY_EPSILON = 1E-8;
	/**
	 * Default score for regulators not in the given file
	 */
	static final double DEFAULTREGSCORE = 0.5;

	private String SZDELIM;
	

	/**
	 * When calling this constructor, the caller must filter the data set if
	 * bfilterbinding is set to true. The call is as follows: if(bfilterbinding)
	 * { theDataSet = new DREM_DataSet(theDataSet.filtergenesgeneral(
	 * bindingData.bbindingdata,bindingData.nbinding, true) , theDataSet.tga); }
	 * 
	 * @throws IOException
	 */
	public RegulatorBindingData(String tfBindingDataFile, List<String> genelist, 
			String szdelim, String regScoreFile, String xrefFile) throws IOException {

		SZDELIM = szdelim;
		geneID2RegBinding = new HashMap<String, BindingGeneRec>();
		existingBindingValuesUnsorted = new HashSet<Double>();
		numberRegs = 0;
		numberGenes = genelist.size();
		regNames = new String[0];
		regTypes = new int[0];
		int offset = 0;
		
		String dataFiles = tfBindingDataFile;

			if (dataFiles != null && !dataFiles.equals("")) {
				BufferedReader br = null;
				try {            //���ļ�
					br = new BufferedReader(new InputStreamReader(
							new GZIPInputStream(new FileInputStream(dataFiles))));
				} catch (IOException ex) {
					br = new BufferedReader(new FileReader(dataFiles));
				}

				String szLine = br.readLine();
				StringTokenizer st = new StringTokenizer(szLine, "\t");
				String szh1 = "";

				int numType;
				if (szLine == null) {
					throw new IllegalArgumentException("Empty regulator-gene interaction input file found!");
				} else if (szLine.startsWith("\t")) {
					numType = st.countTokens();
				} else {
					numType = st.countTokens() - 1;
					szh1 = st.nextToken();
				}

				// Read the top row of the file in      //������
				String[] tempRegNames = new String[numType];
				for (int nRegIndex = 0; st.hasMoreTokens(); nRegIndex++) {
					tempRegNames[nRegIndex] = st.nextToken();
				}
				parseThreeColFormat(br, offset, 0);
				offset = numberRegs;
			}
			
		
		//////////////////////////////////////////////

		existingBindingValuesUnsorted.add(new Double(0));
		
			gene2RegBinding = new double[genelist.size()][];
			gene2RegBindingIndex = new int[genelist.size()][];

			for (int nrow = 0; nrow < genelist.size(); nrow++) {
				Object obj = getBindingObject(genelist.get(nrow));

				if (obj != null) {
					BindingGeneRec theBindingGeneRec = (BindingGeneRec) obj;
					gene2RegBinding[nrow] = theBindingGeneRec.gene2RegulatorBindingRec;
					gene2RegBindingIndex[nrow] = theBindingGeneRec.gene2RegulatorBindingRecIndex;
					for (int nindex = 0; nindex < theBindingGeneRec.gene2RegulatorBindingRec.length; nindex++) {
						existingBindingValuesUnsorted.add(new Double(theBindingGeneRec.gene2RegulatorBindingRec[nindex]));
					}
				} else {
					gene2RegBinding[nrow] = new double[0];
					gene2RegBindingIndex[nrow] = new int[0];
				}
			}
		

		// Initialize and create the regulator-wise arrays and signed arrays
		reg2GeneBindingIndex = new int[numberRegs][];
		reg2GeneBinding = new double[numberRegs][];
		gene2RegBindingSigned = new int[gene2RegBinding.length][];
		reg2GeneBindingSigned = new int[numberRegs][];

		makeRegulatorIndex(gene2RegBinding, gene2RegBindingIndex, 
				reg2GeneBinding, reg2GeneBindingIndex);
		
		makeSigned(gene2RegBinding, gene2RegBindingSigned);
		makeSigned(reg2GeneBinding, reg2GeneBindingSigned);

		// Construct the sorted binding values array
		int nel = 0;
		existingBindingValuesSorted = new double[existingBindingValuesUnsorted
				.size()];
		for (Double d : existingBindingValuesUnsorted) {
			existingBindingValuesSorted[nel] = d.doubleValue();
			nel++;
		}
		Arrays.sort(existingBindingValuesSorted);

		// Contruct the sorted and unsorted signed binding value objects
		signedBindingValuesUnsorted = new HashSet<Integer>();
		for (Double d : existingBindingValuesUnsorted) {
			signedBindingValuesUnsorted.add(new Integer((int) Math.signum(d)));
		}
		signedBindingValuesSorted = new int[signedBindingValuesUnsorted.size()];
		int i = 0;
		for (Integer integer : signedBindingValuesUnsorted) {
			signedBindingValuesSorted[i] = integer.intValue();
			i++;
		}
		Arrays.sort(signedBindingValuesSorted);

		// Construct the regulator synonyms hash table
		HashSet<String> hsRegNames = new HashSet<String>();
		regSyns = new HashMap<String, HashSet<String>>();
		if (xrefFile != null && !xrefFile.equals("")) {
			for (int k = 0; k < regNames.length; k++) {
				hsRegNames.add(regNames[k].toUpperCase());
				regSyns.put(regNames[k].toUpperCase(), new HashSet<String>());
			}
			buildxref(hsRegNames, regSyns, xrefFile);
		}

		// Create a map from regulator name to binding data row
		regRowMap = new HashMap<String, Integer>();
		for (int j = 0; j < regNames.length; j++)
			regRowMap.put(regNames[j], j);

		// If we have a regulator score file create the score array
		
		if (regScoreFile != null && !regScoreFile.equals("")) {
			regPriors = new double[numberRegs];
			boolean[] scored = new boolean[numberRegs];
			double maxPrior = 0;
			BufferedReader br = null;
			try {
				br = new BufferedReader(new InputStreamReader(
						new GZIPInputStream(new FileInputStream(regScoreFile))));
			} catch (IOException ex) {
				br = new BufferedReader(new FileReader(regScoreFile));
			}
			String line;
			while ((line = br.readLine()) != null) {
				StringTokenizer st = new StringTokenizer(line, "\t");
				if (st.countTokens() != 2)
					throw new IOException("The regulator score file should"
							+ " have two columns: <regulator>\t<score>");
				String reg = st.nextToken();
				double score = Double.parseDouble(st.nextToken());
				Integer row = regRowMap.get(reg);
				if (row == null) {
					if (regSyns.containsKey(reg)) {
						for (String syn : regSyns.get(reg)) {
							row = regRowMap.get(syn);
							if (row != null)
								break;
						}
					}
					if (row == null) {
						throw new IOException(
								"All regulators in the regulator score "
										+ "should either be in or have a synonym in the regulator"
										+ " binding data file.");
					}
				}
				regPriors[row] = score;
				scored[row] = true;
				maxPrior = Math.max(maxPrior, score);
			}
			// Activity priors must be in the range [0,1]. Because the absolute
			// value of the binding values was used, no prior will be < 0.
			// However, if any value is > 1, the priors must be normalized.
			if (maxPrior > 1) {
				for (int j = 0; j < regPriors.length; j++)
					regPriors[j] /= maxPrior;
			}
			// To avoid dividing by 0 when a TF activity prior is 1, adjust
			// all priors by ACTIVITY_EPSILON. Priors are allowed to be 0.
			for (int ntf = 0; ntf < regPriors.length; ntf++) {
				if (scored[ntf]) {
					if (regPriors[ntf] > ACTIVITY_EPSILON) {
						regPriors[ntf] -= ACTIVITY_EPSILON;
					} else {
						regPriors[ntf] = 0;
					}
				} else {
					regPriors[ntf] = DEFAULTREGSCORE;
				}
			}
		}
		

		// Construct the set of regulator types
		existingRegTypes = new HashSet<Integer>();
		for (int j = 0; j < regTypes.length; j++) {
			existingRegTypes.add(regTypes[j]);
		}

		// Construct the regulator to type map
		regTypeMap = new HashMap<String, Integer>();
		for (int j = 0; j < regTypes.length; j++) {
			regTypeMap.put(regNames[j], regTypes[j]);
		}
	}

	/**
	 * This constructor build a binding data out of another one based on a
	 * boolean array, keeping the true entries. This is used to create the train
	 * and test data.
	 * 
	 * @param total
	 *            The original RegulatorBindingData to build from.
	 * @param keep
	 *            The boolean array with true entries meaning keep this index.
	 * @param keepCount
	 *            The number of total entries to be kept.
	 */
	public RegulatorBindingData(RegulatorBindingData total, boolean[] keep,
			int keepCount, boolean removeGenes) {
		if (removeGenes) {
			// Copy in the binding data that we want to keep
			gene2RegBinding = new double[keepCount][];
			gene2RegBindingIndex = new int[keepCount][];
			existingBindingValuesUnsorted = new HashSet<Double>();
				int keepIndex = 0;
				for (int i = 0; i < keep.length; i++) {
					if (keep[i]) {
						gene2RegBinding[keepIndex] = new double[total.gene2RegBinding[i].length];
						gene2RegBindingIndex[keepIndex] = new int[total.gene2RegBindingIndex[i].length];
						for (int j = 0; j < total.gene2RegBinding[i].length; j++) {
							gene2RegBinding[keepIndex][j] = total.gene2RegBinding[i][j];
							gene2RegBindingIndex[keepIndex][j] = total.gene2RegBindingIndex[i][j];
							existingBindingValuesUnsorted.add(gene2RegBinding[keepIndex][j]);
						}
						keepIndex++;
					}
				}
			// Construct the remaining fields
			numberRegs = total.numberRegs;
			reg2GeneBinding = new double[numberRegs][];
			reg2GeneBindingIndex = new int[numberRegs][];
			gene2RegBindingSigned = new int[keepCount][];
			reg2GeneBindingSigned = new int[numberRegs][];
			makeRegulatorIndex(gene2RegBinding, gene2RegBindingIndex, reg2GeneBinding,
						reg2GeneBindingIndex);
			makeSigned(gene2RegBinding, gene2RegBindingSigned);
			makeSigned(reg2GeneBinding, reg2GeneBindingSigned);

			regNames = new String[numberRegs];
			regTypes = new int[numberRegs];
			if (total.regPriors != null) {
				regPriors = new double[numberRegs];
				for (int i = 0; i < numberRegs; i++)
					regPriors[i] = total.regPriors[i];
			}
			existingRegTypes = new HashSet<Integer>();
			regRowMap = new HashMap<String, Integer>();
			for (int i = 0; i < numberRegs; i++) {
				regNames[i] = total.regNames[i];
				regTypes[i] = total.regTypes[i];
				existingRegTypes.add(regTypes[i]);
				regRowMap.put(regNames[i], i);
			}
			existingBindingValuesSorted = new double[existingBindingValuesUnsorted
					.size()];
			int k = 0;
			for (Double d : existingBindingValuesUnsorted) {
				existingBindingValuesSorted[k] = d;
				k++;
			}
			Arrays.sort(existingBindingValuesSorted);

			signedBindingValuesUnsorted = new HashSet<Integer>();
			for (Double d : existingBindingValuesUnsorted) {
				signedBindingValuesUnsorted.add((int) Math.signum(d));
			}

			signedBindingValuesSorted = new int[signedBindingValuesUnsorted
					.size()];
			int i = 0;
			for (Integer integer : signedBindingValuesUnsorted) {
				signedBindingValuesSorted[i] = integer.intValue();
				i++;
			}
			Arrays.sort(signedBindingValuesSorted);

			regSyns = total.regSyns;
			regTypeMap = total.regTypeMap;

			// Construct the max,union of the temporal data
			numberGenes = keepCount;
			
		} else { // Remove regulators
			System.out.println("FILTERING MIRNA");
			for (int i = 0; i < keep.length; i++) {
				System.out.println("Reg: " + total.regNames[i]);
				System.out.println("\tKeeping? " + keep[i]);
			}

			numberGenes = total.numberGenes;
			existingBindingValuesUnsorted = new HashSet<Double>();
			int[] newRegIndex = new int[keep.length];
			int newIndex = 0;
			for (int i = 0; i < newRegIndex.length; i++) {
				if (keep[i]) {
					newRegIndex[i] = newIndex;
					newIndex++;

					System.out.println("New index for " + i + " is "
							+ newRegIndex[i]);
				} else {
					newRegIndex[i] = -1;
				}
			}
			numberRegs = newIndex;
			gene2RegBinding = new double[numberGenes][];
			gene2RegBindingIndex = new int[numberGenes][];
				for (int j = 0; j < total.gene2RegBinding.length; j++) {
					int count = 0;
					for (int k = 0; k < total.gene2RegBinding[j].length; k++) {
						if (keep[total.gene2RegBindingIndex[j][k]]) {
							count++;
						}
					}
					gene2RegBinding[j] = new double[count];
					gene2RegBindingIndex[j] = new int[count];
					count = 0;
					for (int k = 0; k < total.gene2RegBinding[j].length; k++) {
						if (keep[total.gene2RegBindingIndex[j][k]]) {
							gene2RegBinding[j][count] = total.gene2RegBinding[j][k];
							gene2RegBindingIndex[j][count] = newRegIndex[total.gene2RegBindingIndex[j][k]];
							existingBindingValuesUnsorted.add(total.gene2RegBinding[j][k]);
							count++;
						}
					}
				}

			for (int i = 0; i < gene2RegBinding.length; i++) {
				System.out.println("Gene " + i);
				for (int j = 0; j < gene2RegBinding[i].length; j++) {
					System.out.println("\tReg " + gene2RegBindingIndex[i][j]);
				}
			}
			// Construct the remaining fields
			reg2GeneBinding = new double[numberRegs][];
			reg2GeneBindingIndex = new int[numberRegs][];
			gene2RegBindingSigned = new int[numberGenes][];
			reg2GeneBindingSigned = new int[numberRegs][];
			makeRegulatorIndex(gene2RegBinding, gene2RegBindingIndex, 
					reg2GeneBinding, reg2GeneBindingIndex);
			makeSigned(gene2RegBinding, gene2RegBindingSigned);
			makeSigned(reg2GeneBinding, reg2GeneBindingSigned);

			regNames = new String[numberRegs];
			regTypes = new int[numberRegs];
			if (total.regPriors != null) {
				regPriors = new double[numberRegs];
				for (int i = 0; i < total.numberRegs; i++) {
					if (keep[i])
						regPriors[newRegIndex[i]] = total.regPriors[i];
				}
			}
			existingRegTypes = new HashSet<Integer>();
			regRowMap = new HashMap<String, Integer>();
			for (int i = 0; i < total.numberRegs; i++) {
				if (keep[i]) {
					regNames[newRegIndex[i]] = total.regNames[i];
					regTypes[newRegIndex[i]] = total.regTypes[i];
					existingRegTypes.add(regTypes[newRegIndex[i]]);
					regRowMap.put(regNames[newRegIndex[i]], newRegIndex[i]);
				}
			}
			existingBindingValuesSorted = new double[existingBindingValuesUnsorted
					.size()];
			int k = 0;
			for (Double d : existingBindingValuesUnsorted) {
				existingBindingValuesSorted[k] = d;
				k++;
			}
			Arrays.sort(existingBindingValuesSorted);

			signedBindingValuesUnsorted = new HashSet<Integer>();
			for (Double d : existingBindingValuesUnsorted) {
				signedBindingValuesUnsorted.add((int) Math.signum(d));
			}

			signedBindingValuesSorted = new int[signedBindingValuesUnsorted.size()];
			int i = 0;
			for (Integer integer : signedBindingValuesUnsorted) {
				signedBindingValuesSorted[i] = integer.intValue();
				i++;
			}
			Arrays.sort(signedBindingValuesSorted);

			regSyns = total.regSyns;
			regTypeMap = total.regTypeMap;

			// Construct the max,union of the temporal data
			numberGenes = total.numberGenes;
			
			System.out.println("FILTERING FINISHED");
			System.out.println("Number regs: " + numberRegs);
			for (i = 0; i < regNames.length; i++) {
				System.out.println("Reg: " + regNames[i]);
				System.out.println("Binding for reg " + i + ":");
				for (int j = 0; j < reg2GeneBinding[i].length; j++) {
					System.out.println("\tBinds " + reg2GeneBindingIndex[i][j]);
				}
			}
		}
	}

	/**
	 * Record of index and regulatory value
	 */
	private static class RegulatorRec {
		int nRegulatorIndex;
		double nRegVal;

		RegulatorRec(int nRegulatorIndex, double nRegVal) {
			this.nRegulatorIndex = nRegulatorIndex;
			this.nRegVal = nRegVal;
		}
	}

	/**
	 * Comparator object sorts strictly based on nRegulatorIndex
	 */
	private class RegulatorRecCompare implements Comparator<RegulatorRec> {
		public int compare(RegulatorRec regr1, RegulatorRec regr2) {
			if (regr1.nRegulatorIndex < regr2.nRegulatorIndex) {
				return -1;
			} else if (regr1.nRegulatorIndex > regr2.nRegulatorIndex) {
				return 1;
			} else {
				return 0;
			}
		}
	}

	/**
	 * Reads in the regulator-gene interaction data currently in the format
	 * regulator, gene, and if specified interaction value where each column is
	 * delimitted by tabs
	 */
	private void parseThreeColFormat(BufferedReader br, int offset, int type)
			throws IOException {
		HashMap<String, ArrayList<RegulatorRec>> htGeneTimeToRegulatorArray = new HashMap<String, ArrayList<RegulatorRec>>();//����+ʱ����Ӧ��tf����
		HashMap<String, Integer> htRegulatorToInteger = new HashMap<String, Integer>();  //TF��Ӧgene
		// Start at offset so mirna data will have differenet indicies from tfs
		int nReg = offset;
		String szLine;
		while ((szLine = br.readLine()) != null) {
			StringTokenizer st = new StringTokenizer(szLine, "\t");
			String szRegulator = st.nextToken();
			String szGene = st.nextToken();
			double ninput;
			if (st.hasMoreTokens()) {
				String szToken = st.nextToken();
				try {
					ninput = Double.parseDouble(szToken);
				} catch (NumberFormatException nfex) {
					throw new IllegalArgumentException(szToken + " is not a"
							+ " valid score for a regulator-gene interaction");
				}
			} else {
				ninput = 1.0;
			}
			
			
			//if(ninput != 0){
				Integer objInt = htRegulatorToInteger.get(szRegulator);
				int nCurrRegulator;
				if (objInt == null) {
					nCurrRegulator = nReg;
					htRegulatorToInteger.put(szRegulator, new Integer(nReg));
					nReg++;
				} else {
					nCurrRegulator = ((Integer) objInt).intValue();
				}
				ArrayList<RegulatorRec> al = htGeneTimeToRegulatorArray.get(szGene);
				if (al == null) {
					al = new ArrayList<RegulatorRec>();
				}
				al.add(new RegulatorRec(nCurrRegulator, ninput));
				htGeneTimeToRegulatorArray.put(szGene, al);
			//}
		}
		br.close();
		
		if (offset > 0) {
			// If we are offset, then we need to create new larger arrays
			int numberNewRegulators = htRegulatorToInteger.size();
			String[] totalRegulatorNames = new String[offset + numberNewRegulators];
			for (int i = 0; i < offset; i++) totalRegulatorNames[i] = regNames[i];
			regNames = totalRegulatorNames;
			int[] totalRegulatorTypes = new int[offset + numberNewRegulators];
			for (int i = 0; i < offset; i++) totalRegulatorTypes[i] = regTypes[i];
			regTypes = totalRegulatorTypes;
			numberRegs = regNames.length;
		} else {
			numberRegs = htRegulatorToInteger.size();
			regNames = new String[numberRegs];
			regTypes = new int[numberRegs];
		}
		
		Iterator<String> itrRegulatorSet = htRegulatorToInteger.keySet().iterator();
		while (itrRegulatorSet.hasNext()) {
			String szRegulator = itrRegulatorSet.next();
			int index = ((Integer) htRegulatorToInteger.get(szRegulator)).intValue();
			regNames[index] = szRegulator;
			regTypes[index] = type;
		}

		Iterator<String> itrGeneSet = htGeneTimeToRegulatorArray.keySet().iterator();
		while (itrGeneSet.hasNext()) {
			String gat = itrGeneSet.next();
			ArrayList<RegulatorRec> al = htGeneTimeToRegulatorArray.get(gat);
			int numnonzero = al.size();

			RegulatorRec[] thePairsRecs = new RegulatorRec[numnonzero];
			for (int nbit = 0; nbit < numnonzero; nbit++) {
				thePairsRecs[nbit] = (RegulatorRec) al.get(nbit);
			}
			Arrays.sort(thePairsRecs, new RegulatorRecCompare());
			int[] nonzeroindex = new int[numnonzero];
			double[] nonzerovals = new double[numnonzero];
			for (int i = 0; i < numnonzero; i++) {
				RegulatorRec theRegulatorRec = thePairsRecs[i];
				nonzeroindex[i] = theRegulatorRec.nRegulatorIndex;
				nonzerovals[i] = theRegulatorRec.nRegVal;
			}
			loadBinding(gat, nonzeroindex, nonzerovals, numnonzero);
		}
	}

	class GeneAndTimepoint {
		String gene;
		int timepoint;

		public GeneAndTimepoint(String gene, int timepoint) {
			this.gene = gene;
			this.timepoint = timepoint;
		}

		public int hashCode() {
			return gene.hashCode() + timepoint * 23;
		}
	}

	/**
	 * Reads in the regulator-gene input data assuming the header is already
	 * read and the data is in grid format
	 */
	private void parseGridFormat(BufferedReader br, int offset, int regCount)
			throws IOException {
		int[] nonzeroindex = new int[regCount];
		double[] nonzerovals = new double[regCount];
		String szLine;
		StringTokenizer st;

		while ((szLine = br.readLine()) != null) {
			int numnonzero = 0;

			if (!szLine.trim().equals("")) {
				st = new StringTokenizer(szLine, "\t");
				int numTokens = st.countTokens();
				if (regCount != (numTokens - 1)) {
					throw new IllegalArgumentException("Found a line with "
							+ (numTokens - 1) + " entries, expecting "
							+ numberRegs);
				}
				String szid = st.nextToken();

				for (int nindex = 0; nindex < regCount; nindex++) {
					String sznum = st.nextToken();
					try {
						double tempval = Double.parseDouble(sznum);
						if (tempval != 0) {
							nonzeroindex[numnonzero] = offset + nindex;
							nonzerovals[numnonzero] = tempval;
							numnonzero++;
						}
					} catch (NumberFormatException ex) {
						if ((numberRegs == 2) && (nindex == 0)) {
							throw new IllegalArgumentException(
									"If regulator-gene data is in column format,"
											+ " then the first two columns must have the "
											+ "headers 'Regulator Type' and 'Gene'");
						} else {
							throw new IllegalArgumentException(sznum
									+ " is not a valid value for a "
									+ "regulator-gene interaction!");
						}
					}
				}
				// storing each vector of binding values in hashtable with gene
				// identifier
				loadBinding(szid, nonzeroindex, nonzerovals, numnonzero);
			}
		}
		br.close();
	}

	/**
	 * A recording contain an array of interaction values and another array
	 * containing the indices they correspond to
	 */
	static class BindingGeneRec {
		double[] gene2RegulatorBindingRec;
		int[] gene2RegulatorBindingRecIndex;
	}

	/**
	 * Stores into geneID2RegulatorBinding the mapping of the gene szid to its
	 * non-zero regulator-gene interactions given by nonzeroindex and
	 * nonzerovals
	 */
	public void loadBinding(String szid, int[] nonzeroindex,
			double[] nonzerovals, int numnonzero) {
		String szfull = szid.toUpperCase(Locale.ENGLISH);
		StringTokenizer stIDs = new StringTokenizer(szfull, SZDELIM);

		BindingGeneRec rec = new BindingGeneRec();
		rec.gene2RegulatorBindingRecIndex = new int[numnonzero];
		rec.gene2RegulatorBindingRec = new double[numnonzero];
				for (int j = 0; j < numnonzero; j++) {
					rec.gene2RegulatorBindingRecIndex[j] = nonzeroindex[j];
					rec.gene2RegulatorBindingRec[j] = nonzerovals[j];
				}

		while (stIDs.hasMoreTokens()) {
			// union if gene matches multiple hits in binding file, takes last
			// one seen if multiple non-zero
			String sztoken = stIDs.nextToken();
			BindingGeneRec currec = geneID2RegBinding.get(sztoken);

			if (currec != null) {
				rec = mergeArrays(rec.gene2RegulatorBindingRec,
						rec.gene2RegulatorBindingRecIndex,
						currec.gene2RegulatorBindingRec,
						currec.gene2RegulatorBindingRecIndex);
			}
			if (rec.gene2RegulatorBindingRec == null)
				System.out.println("ERROR binding array is null");
			if (rec.gene2RegulatorBindingRecIndex == null)
				System.out.println("ERROR index array is null");
			geneID2RegBinding.put(sztoken, rec);

			StringTokenizer stu = new StringTokenizer(sztoken, "_");
			if (stu.countTokens() > 1) {
				String szfirsttoken = stu.nextToken();
				currec = geneID2RegBinding.get(szfirsttoken);
				if (currec != null) {
					rec = mergeArrays(rec.gene2RegulatorBindingRec,
							rec.gene2RegulatorBindingRecIndex,
							currec.gene2RegulatorBindingRec,
							currec.gene2RegulatorBindingRecIndex);
				}
				if (rec.gene2RegulatorBindingRec == null)
					System.out.println("ERROR array is null");
				if (rec.gene2RegulatorBindingRecIndex == null)
					System.out.println("ERROR index array is null");
				geneID2RegBinding.put(szfirsttoken, rec);
			}
		}
	}

	/**
	 * Returns an entry of geneID2RegulatorBinding corresponding to szprobename
	 * or szgenename
	 */
	public BindingGeneRec getBindingObject(String szgenename) {

		StringTokenizer st = new StringTokenizer(szgenename, SZDELIM);
		String sztoken;
		BindingGeneRec bgr = null;

		st = new StringTokenizer(szgenename, SZDELIM);
		while ((st.hasMoreTokens()) && (bgr == null)) {
			sztoken = st.nextToken();
			bgr = geneID2RegBinding.get(sztoken);
		}

		return bgr;
	}

	/**
	 * Assume geneindex1 and geneindex2 are sorted arrays Returns a
	 * BindingGeneRec where bindingpvalGeneindex has the union of geneindex1 and
	 * geneindex2 and bindingpvalGene has the corresponding values using the
	 * maximum to resolve disagreements
	 */
	private BindingGeneRec mergeArrays(double[] geneval1, int[] geneindex1,
			double[] geneval2, int[] geneindex2) {
		BindingGeneRec theBindingGeneRec = new BindingGeneRec();
			int nindex1 = 0;
			int nindex2 = 0;
			int nmatch = 0;
			while ((nindex1 < geneindex1.length) && (nindex2 < geneindex2.length)) {
				if (geneindex1[nindex1] == geneindex2[nindex2]) {
					nmatch++;
					nindex1++;
					nindex2++;
				} else if (geneindex1[nindex1] < geneindex2[nindex2]) {
					nindex1++;
				} else if (geneindex2[nindex2] < geneindex1[nindex1]) {
					nindex2++;
				}
			}
			int nmergecount = geneindex1.length + geneindex2.length - nmatch;
			theBindingGeneRec.gene2RegulatorBindingRec = new double[nmergecount];
			theBindingGeneRec.gene2RegulatorBindingRecIndex = new int[nmergecount];
			int nmergeindex = 0;
			nindex1 = 0;
			nindex2 = 0;
			while ((nindex1 < geneindex1.length)
					&& (nindex2 < geneindex2.length)) {
				if (geneindex1[nindex1] == geneindex2[nindex2]) {
					theBindingGeneRec.gene2RegulatorBindingRec[nmergeindex] = Math
							.max(geneval1[nindex1], geneval2[nindex2]);
					theBindingGeneRec.gene2RegulatorBindingRecIndex[nmergeindex] = geneindex2[nindex2];
					nindex1++;
					nindex2++;
				} else if (geneindex1[nindex1] < geneindex2[nindex2]) {
					theBindingGeneRec.gene2RegulatorBindingRec[nmergeindex] = geneval1[nindex1];
					theBindingGeneRec.gene2RegulatorBindingRecIndex[nmergeindex] = geneindex1[nindex1];
					nindex1++;
				} else if (geneindex2[nindex2] < geneindex1[nindex1]) {
					theBindingGeneRec.gene2RegulatorBindingRec[nmergeindex] = geneval2[nindex2];
					theBindingGeneRec.gene2RegulatorBindingRecIndex[nmergeindex] = geneindex2[nindex2];
					nindex2++;
				}
				nmergeindex++;
			}

			while (nindex2 < geneindex2.length) {
				theBindingGeneRec.gene2RegulatorBindingRec[nmergeindex] = geneval2[nindex2];
				theBindingGeneRec.gene2RegulatorBindingRecIndex[nmergeindex] = geneindex2[nindex2];
				nmergeindex++;
				nindex2++;
			}
			while (nindex1 < geneindex1.length) {
				theBindingGeneRec.gene2RegulatorBindingRec[nmergeindex] = geneval1[nindex1];
				theBindingGeneRec.gene2RegulatorBindingRecIndex[nmergeindex] = geneindex1[nindex1];
				nmergeindex++;
				nindex1++;
			}
		return theBindingGeneRec;
	}

	/**
	 * Generates valReg and valRegIndex based on val and valIndex so that the
	 * rows correspond to each regulator and each entry in a row valRegIndex
	 * corresponds to an index of a gene the regulator regulates and the
	 * corresponding entry in valReg corresponds to the interaction value
	 */
	public void makeRegulatorIndex(double[][] val, int[][] valIndex,
			double[][] valReg, int[][] valRegIndex) {
		int[] counts = new int[valReg.length];
		int sum = 0;
		for (int i = 0; i < valIndex.length; i++) {
			if (valIndex[i] == null)
				System.out.println("Null index is: " + i);
			for (int j = 0; j < valIndex[i].length; j++) {
				counts[valIndex[i][j]]++;
				sum++;
			}
		}

		for (int i = 0; i < valRegIndex.length; i++) {
			valReg[i] = new double[counts[i]];
			valRegIndex[i] = new int[counts[i]];
		}

		// reusing counts as number found so far
		for (int i = 0; i < counts.length; i++) {
			counts[i] = 0;
		}

		for (int ngene = 0; ngene < valIndex.length; ngene++) {
			for (int nregindex = 0; nregindex < valIndex[ngene].length; nregindex++) {
				int nReg = valIndex[ngene][nregindex];
				int nRegindex = counts[valIndex[ngene][nregindex]];
				valReg[nReg][nRegindex] = val[ngene][nregindex];
				valRegIndex[nReg][nRegindex] = ngene;
				counts[nReg]++;
			}
		}
	}

	/**
	 * Takes in a values array and fills in the Signed array with the signs of
	 * the entries in values {-1,0,1}
	 */
	public void makeSigned(double[][] values, int[][] valuesSigned) {
			//valuesSigned = new int[values.length][];
			for (int i = 0; i < values.length; i++) {
				valuesSigned[i] = new int[values[i].length];
				for (int j = 0; j < values[i].length; j++) {
					valuesSigned[i][j] = (int) Math.signum(values[i][j]);
				}
			}
	}

	/**
	 *Builds a cross reference mapping which as keys is alternative identifiers
	 * that map to a subset of genes in htgenes
	 */
	public void buildxref(HashSet<String> htregs,
			HashMap<String, HashSet<String>> htxref, String szxrefval)
			throws FileNotFoundException, IOException {
		if ((szxrefval != null) && (!szxrefval.equals(""))) {
			// we have cross references from which to build
			BufferedReader br;
			try {
				// first tries gzip format, if that fails then does regular
				br = new BufferedReader(new InputStreamReader(
						new GZIPInputStream(new FileInputStream(szxrefval))));
			} catch (IOException ex) {
				br = new BufferedReader(new FileReader(szxrefval));
			}

			String szLine;
			while ((szLine = br.readLine()) != null) {
				// htfoundgenes contains genes in the htgenes also on the line
				HashSet<String> htfoundgenes = new HashSet<String>(3);
				// htallsyn has all genes on the line
				HashSet<String> htallsyn = new HashSet<String>();
				// entries can be delimited by a tab, semicolon, pipe, comma, or
				// double quote
				StringTokenizer st = new StringTokenizer(szLine, "\"\t;|,");
				while (st.hasMoreTokens()) {
					String sztoken = st.nextToken().trim().toUpperCase();

					// adding to synonym list
					htallsyn.add(sztoken);

					if (htregs.contains(sztoken)) {
						// storing known genes
						htfoundgenes.add(sztoken);
					}
				}
				if (!htfoundgenes.isEmpty()) {
					// foundgenes not empty implies htallsynitr not empty
					Iterator<String> htallsynitr = htallsyn.iterator();
					while (htallsynitr.hasNext()) {
						String szsyn = htallsynitr.next();
						HashSet<String> hssyngenes = htxref.get(szsyn);

						if (hssyngenes == null) {
							hssyngenes = new HashSet<String>();
						}
						// mapping synonym to all terms
						hssyngenes.addAll(htfoundgenes);
						htxref.put(szsyn, hssyngenes);
					}
				}
			}
			br.close();
		}
	}
	/*
	public void adjustBindingData(HashMap<String, Integer> reg2DataSetIndex,
			BSM_DataSet theDataSet, bsm.core.DataSetCore miRNADataSet,
			boolean scaleMIRNAExp, boolean scaleTFExp,
			double miRNAScalingFactor, double minTFExp) {
		for (int time = 0; time < theDataSet.numcols; time++) {
			for (int geneIndex = 0; geneIndex < gene2RegBinding[time].length; geneIndex++) {
				for (int numRegulator = 0; numRegulator < gene2RegBindingIndex[time][geneIndex].length; numRegulator++) {
					double bindingValue = gene2RegBinding[time][geneIndex][numRegulator];
					int bindingIndexValue = gene2RegBindingIndex[time][geneIndex][numRegulator];
					Integer dataSetIndex = reg2DataSetIndex
							.get(regNames[bindingIndexValue].toUpperCase());
					if (bindingValue == 0) {
						gene2RegBinding[time][geneIndex][numRegulator] = 0;
					} else if (regTypes[bindingIndexValue] == RegulatorBindingData.MIRNA
							&& dataSetIndex != null
							&& scaleMIRNAExp
							&& time < timepoints - 1) {
						gene2RegBinding[time][geneIndex][numRegulator] = logitShiftZeroNegative(
								miRNADataSet.data[dataSetIndex][time + 1]
										- miRNADataSet.data[dataSetIndex][time],
								miRNAScalingFactor);
					} else if (regTypes[bindingIndexValue] == RegulatorBindingData.TF
							&& dataSetIndex != null
							&& scaleTFExp
							&& time < timepoints - 1) {
						double interValue = logitShiftZeroNegative(theDataSet.data[dataSetIndex][time + 1]- theDataSet.data[dataSetIndex][time],miRNAScalingFactor);
						if (Math.abs(interValue) < minTFExp)
							interValue = minTFExp * Math.signum(interValue);
						gene2RegBinding[time][geneIndex][numRegulator] = interValue;
					} else {
						gene2RegBinding[time][geneIndex][numRegulator] = bindingValue;
					}

					
					 //Scale the activity values by the regulator priors.
					 
					if (regPriors != null)
						gene2RegBinding[time][geneIndex][numRegulator] *= regPriors[numRegulator];
				}
			}
			makeRegulatorIndex(gene2RegBinding[time],
					gene2RegBindingIndex[time], reg2GeneBinding[time],
					reg2GeneBindingIndex[time]);
		}
	}
	*/
	public double logit(double x, double w) {
		return (1 / (1 + Math.exp(-w * x)));
	}

	/**
	 * A modified version of the logit function that shifts the logit function
	 * to 0, s.t. when x is 0 the function returns 0, instead of 0.5 normally.
	 * Also instead of [0,1] the function outputs values in the range [-1,1] to
	 * allow for negative influence. As with a normal logit function the weight
	 * w can be used to control the steepness of the function. Currently, it is
	 * assumed that the function is only called for entries that are supposed to
	 * be regulating, that means if there is no regulatory relationship than
	 * logitShiftZeroNegative should always return 0, but that has to be
	 * controlled before.
	 */

	public double logitShiftZeroNegative(double x, double w) {
		if (x == 0) {
			return (0);
		} else {
			return (Math.signum(x) * ((logit(Math.abs(x), w) - 0.5) * 2));
		}
	}
	
	public void deDuplication() {
		int[][] gene2RegDedupliIndex = new int[numberGenes][];
		double[][] gene2RegDedupli = new double[numberGenes][];
		int[][] reg2GeneDedupliIndex = new int[numberRegs][];
		double[][] reg2GeneDedupli = new double[numberRegs][];
		
		for(int i=0;i<numberGenes;i++){
			int[] tfindex = gene2RegBindingIndex[i];
			double[] tfvalue = gene2RegBinding[i];
			Set<Integer> set = new HashSet<Integer>();
			for(int j:tfindex) set.add(j);
			
			gene2RegDedupliIndex[i] = new int[set.size()];
			gene2RegDedupli[i] = new double[set.size()];
			int count = 0;
			for(int tf:set){
				gene2RegDedupliIndex[i][count] = tf;
				for(int j=0;j<tfindex.length;j++){
					if(tfindex[j] == tf){
						if(Math.abs(gene2RegDedupli[i][count]) < Math.abs(tfvalue[j])){
							gene2RegDedupli[i][count] = tfvalue[j];
						}
					}
				}
				count++;
			}
		}
		for(int i=0;i<numberRegs;i++){
			int[] tgindex = reg2GeneBindingIndex[i];
			double[] tgvalue = reg2GeneBinding[i];
			Set<Integer> set = new HashSet<Integer>();
			for(int j:tgindex) set.add(j);
			
			reg2GeneDedupliIndex[i] = new int[set.size()];
			reg2GeneDedupli[i] = new double[set.size()];
			int count = 0;
			for(int tg:set){
				reg2GeneDedupliIndex[i][count] = tg;
				for(int j=0;j<tgindex.length;j++){
					if(tgindex[j] == tg){
						if(Math.abs(reg2GeneDedupli[i][count]) < Math.abs(tgvalue[j])){
							reg2GeneDedupli[i][count] = tgvalue[j];
						}
					}
				}
				count++;
			}
		}
		this.gene2RegBindingIndex = gene2RegDedupliIndex;
		this.gene2RegBinding = gene2RegDedupli;
		this.reg2GeneBindingIndex = reg2GeneDedupliIndex;
		this.reg2GeneBinding = reg2GeneDedupli;
	}
	
	public void coexpression(BSM_DataSet theds, NormalDistribution Normalcorr, double mincorr,
			HashMap<Integer, List<List<Integer>>> PPIindex, HashMap<Integer, List<Double[]>> PPIvalue) {
		
		HashMap<Integer, List<Integer>> ppiindex = new HashMap<Integer, List<Integer>>();
		HashMap<Integer, List<Double>> ppivalue = new HashMap<Integer, List<Double>>();
		for(int i=0;i<numberGenes;i++){
			for(int j=0;j<gene2RegBindingIndex[i].length;j++) {
				double bingidng1 = gene2RegBinding[i][j];
				int[] geneindex = reg2GeneBindingIndex[gene2RegBindingIndex[i][j]];
				double[] bindingvalue = reg2GeneBinding[gene2RegBindingIndex[i][j]];
				if(geneindex != null && geneindex.length > 1) {
					List<Integer> index = ppiindex.get(i);
					if(index != null && index.size()>0) {
						List<Double> value = ppivalue.get(i);
						for(int p=0;p<geneindex.length;p++) {
							if(geneindex[p] != i && !index.contains(geneindex[p])) {
								index.add(geneindex[p]);
								value.add(bingidng1*bindingvalue[p]);
							}
						}
						ppiindex.put(i, index);
						ppivalue.put(i, value);
					} else {
						index = new ArrayList<Integer>();
						List<Double> value = new ArrayList<Double>();
						for(int p=0;p<geneindex.length;p++) {
							if(geneindex[p] != i) {
								index.add(geneindex[p]);
								value.add(bingidng1*bindingvalue[p]);
							}
						}
						ppiindex.put(i, index);
						ppivalue.put(i, value);
					}
				}
			}
		}
		
		Pearsonr pr = new Pearsonr();
		Correlationpro corpro = new Correlationpro();
		Collection<Integer> cl = ppiindex.keySet();
	    Iterator<Integer> itr = cl.iterator();
	    while (itr.hasNext()) {
	    	int key = itr.next();
	    	double[] exp1 = theds.controlnorm[key];
	    	List<Integer> genes = ppiindex.get(key);
	    	List<Double> bindingvalue = ppivalue.get(key);
	    	if(genes.size() > 0) {
	    		List<Integer> genes2 = new ArrayList<Integer>();
	    		List<Double> value = new ArrayList<Double>();
	    		for(int i=0;i<genes.size();i++) {
	    			double[] exp2 = theds.controlnorm[genes.get(i)];
	    			double corr = pr.cosineSimilarity(exp1, exp2);
	    			if((bindingvalue.get(i) > 0 && corr > mincorr) || (bindingvalue.get(i) < 0 && corr < -mincorr)
	    					|| (bindingvalue.get(i) == 0 && Math.abs(corr) > mincorr)) {
	    				value.add(corpro.getProportion(Normalcorr, corr, mincorr)[1]);
	    				genes2.add(genes.get(i));
	    			}
	    		}
	    		if(value.size() > 1) {
	    			List<List<Integer>> current = PPIindex.get(key);
	    			if(current != null && current.size()>0) {
	    				if(!contain(current, genes2)) {
	    					current.add(genes2);
	    					Double[] weight = new Double[value.size()];
			    			double sum = 0;
			    			for(int i=0;i<value.size();i++) {
			    				weight[i] = Math.exp(value.get(i));
			    				sum += weight[i];
			    			}
			    			for(int i=0;i<weight.length;i++) {
			    				weight[i] /= sum;
			    			}
			    			List<Double[]> currentvalue = PPIvalue.get(key);
			    			currentvalue.add(weight);
			    			PPIindex.put(key, current);
			    			PPIvalue.put(key, currentvalue);
	    				}
	    			} else {
	    				Double[] weight = new Double[value.size()];
		    			double sum=0;
		    			for(int i=0;i<value.size();i++) {
		    				weight[i] = Math.exp(value.get(i));
		    				sum += weight[i];
		    			}
		    			for(int i=0;i<weight.length;i++) {
		    				weight[i] /= sum;
		    			}

		    			List<List<Integer>> list3 = new ArrayList<List<Integer>>();
		    			list3.add(genes2);
		    			List<Double[]> value3 = new ArrayList<Double[]>();
		    			value3.add(weight);
		    			PPIindex.put(key, list3);
		    			PPIvalue.put(key, value3);
	    			}
	    		}
	    	}
	    }
	}
	
	private boolean contain (List<List<Integer>> all, List<Integer> list1) {
		List<Integer> list2 = new ArrayList<Integer>(list1);
		Collections.sort(list2);
		for(int i=0;i<all.size();i++) {
			List<Integer> list = new ArrayList<Integer>(all.get(i));
			Collections.sort(list);
			if(list.toString().equals(list2.toString())) {
				return true;
			}
		}
		return false;
	}

}