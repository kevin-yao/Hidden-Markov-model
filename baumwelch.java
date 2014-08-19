import java.util.*;
import java.util.Map.Entry;
import java.io.*;

public class baumwelch {
	private HashMap<String, HashMap<String, Double>> trans;
	private HashMap<String, HashMap<String, Double>> emit;
	private HashMap<String, Double> prior;
	private ArrayList<String[]> trainingData;
	private Set<String> vocabulary;
	private String[] states = {"PR", "VB", "RB", "NN", "PC", "JJ", "DT", "OT"};
	private static HashMap<Integer, HashMap<String, Double>> alphaRecord;
	private static HashMap<Integer, HashMap<String, Double>> betaRecord;
	private int maxIterNum = 20;
	private double minDifference = 0.1;
	public baumwelch(){
		trans = new HashMap<String, HashMap<String, Double>>();
		emit = new HashMap<String, HashMap<String, Double>>();
		prior = new HashMap<String, Double>();
		alphaRecord = new HashMap<Integer, HashMap<String, Double>>();
		betaRecord = new HashMap<Integer, HashMap<String, Double>>();
	}
	public static void main(String[] args){
		baumwelch be = new baumwelch();
		be.training(args);
		//Iterator<Entry<String, Double>> it =  be.emit.get("JJ").entrySet().iterator();
		//while(it.hasNext()){
		//System.out.println(it.next());
		//}
	}
	public void training(String[] args){
		double preAvgLL;
		double curAvgLL;
		int iterNum = 0;
		double avgLLDiff = 100;
		HashMap<String, Double> updatedPrior;
		HashMap<String, HashMap<String, Double>> updatedTrans;
		HashMap<String, HashMap<String, Double>> updatedEmit;
		init(args);
		preAvgLL = getAvgLL(trainingData);
		System.out.println(preAvgLL);
		while(avgLLDiff > minDifference && iterNum < maxIterNum){
			updatedPrior = updataPrior(trainingData);
			//System.out.println(1);
			updatedTrans = updateTrans(trainingData);
			//System.out.println(2);
			updatedEmit = updateEmit(trainingData);
			//System.out.println(3);
			prior = updatedPrior;
			trans = updatedTrans;
			emit = updatedEmit;
			curAvgLL = getAvgLL(trainingData);
			System.out.println(curAvgLL);
			avgLLDiff = curAvgLL - preAvgLL;
			preAvgLL = curAvgLL;
			iterNum++;
		}
	}
	public double getAlpha(String state, int time, String[] observations){
		double prob = 0;
		HashMap<String, Double>  alphaValue;
		if(time<1){
			System.err.println("time is less than 1");
			System.exit(1);
		}
		if(time == 1){
			prob = Math.log(prior.get(state))+
					Math.log(emit.get(state).get(observations[time-1]));
			if(alphaRecord.get(time) == null){
				alphaValue = new HashMap<String, Double>();
				alphaValue.put(state, prob);
			}else{	
				alphaValue = alphaRecord.get(time);
				alphaValue.put(state, prob);
			}
			alphaRecord.put(time, alphaValue);
			return prob;
		}
		Iterator<Entry<String, HashMap<String, Double>>> it = trans.entrySet().iterator();
		Entry<String, HashMap<String, Double>> entry = it.next();
		String possibleState = entry.getKey();
		double prob1;
		double prob2;
		//If alpha value of (time-1) doesn't exist, call getAlpha to compute it  
		if(alphaRecord.get(time-1) == null || alphaRecord.get(time-1).get(possibleState)==null){
			prob1 = getAlpha(possibleState, time-1, observations)+Math.log(trans.get(possibleState).get(state));
			while(it.hasNext()){
				entry = it.next();
				possibleState = entry.getKey();
				prob2 = getAlpha(possibleState, time-1, observations)+Math.log(trans.get(possibleState).get(state));
				prob1 = Util.logSum(prob1, prob2);
			}
			prob = Math.log(emit.get(state).get(observations[time-1]))+ prob1;	
		}else{//get the value from hashmap
			prob1 = alphaRecord.get(time-1).get(possibleState)+Math.log(trans.get(possibleState).get(state));
			while(it.hasNext()){
				entry = it.next();
				possibleState = entry.getKey();
				prob2 = alphaRecord.get(time-1).get(possibleState)+Math.log(trans.get(possibleState).get(state));
				prob1 = Util.logSum(prob1, prob2);
			}
			prob = Math.log(emit.get(state).get(observations[time-1]))+ prob1;
		}
		//save the alpha value in this time with this state
		if(alphaRecord.get(time) == null ){
			alphaValue = new HashMap<String, Double>();
			alphaValue.put(state, prob);
			alphaRecord.put(time, alphaValue);
		}else{
			alphaValue = alphaRecord.get(time);
			alphaValue.put(state, prob);
			alphaRecord.put(time, alphaValue);
		}
		return prob;
	}

	//return log likelihood
	public double getBeta(String state, int time, String[] observations){
		double prob = 0;
		HashMap<String, Double>  betaValue;
		int  T = observations.length;
		if(time<1){
			System.err.println("time is less than 1");
			System.exit(1);
		}
		if(time == T){
			prob = 0;//log(1) = 0
			if(betaRecord.get(time) == null){
				betaValue = new HashMap<String, Double>();
				betaValue.put(state, prob);
			}else{	
				betaValue = betaRecord.get(time);
				betaValue.put(state, prob);
			}
			betaRecord.put(time, betaValue);
			return prob;
		}
		Iterator<Entry<String, HashMap<String, Double>>> it = trans.entrySet().iterator();
		Entry<String, HashMap<String, Double>> entry = it.next();
		String possibleState = entry.getKey();
		double prob1;
		double prob2;
		//If beta value of (time+1) doesn't exist, call getAlpha to compute it  
		if(betaRecord.get(time+1) == null || betaRecord.get(time+1).get(possibleState)==null){
			prob1 = getBeta(possibleState, time+1, observations)+Math.log(trans.get(state).get(possibleState))
					+ Math.log(emit.get(possibleState).get(observations[time]));
			while(it.hasNext()){
				entry = it.next();
				possibleState = entry.getKey();
				prob2 = getBeta(possibleState, time+1, observations)+Math.log(trans.get(state).get(possibleState))
						+ Math.log(emit.get(possibleState).get(observations[time]));
				prob1 = Util.logSum(prob1, prob2);
			}
		}else{//get the value from hashmap
			prob1 = betaRecord.get(time+1).get(possibleState)+Math.log(trans.get(state).get(possibleState))
					+ Math.log(emit.get(possibleState).get(observations[time]));
			while(it.hasNext()){
				entry = it.next();
				possibleState = entry.getKey();
				prob2 = betaRecord.get(time+1).get(possibleState)+Math.log(trans.get(state).get(possibleState))
						+ Math.log(emit.get(possibleState).get(observations[time]));
				prob1 = Util.logSum(prob1, prob2);
			}
		}
		prob = prob1;
		//save the alpha value in this time with this state
		if(betaRecord.get(time) == null ){
			betaValue = new HashMap<String, Double>();
			betaValue.put(state, prob);
			betaRecord.put(time, betaValue);
		}else if(betaRecord.get(time).get(state)==null){
			betaValue = betaRecord.get(time);
			betaValue.put(state, prob);
			betaRecord.put(time, betaValue);
		}
		return prob;
	}

	public double getGamma(String state, int time, String[] observations){
		double gamma;
		gamma = getAlpha(state, time, observations)+getBeta(state, time, observations);
		double sum=0;
		for(int i=0; i<states.length;i++){
			if(i==0)
				sum = getAlpha(states[i], time, observations)
				+getBeta(states[i], time, observations);
			else
				sum = Util.logSum(sum, getAlpha(states[i], time, observations)+getBeta(states[i], time, observations));
		}
		gamma -= sum;
		return gamma;
	}

	public double getXi(String stateI, String stateJ, int time, String[] observations){
		double xi;
		xi = getGamma(stateI, time, observations)+Math.log(trans.get(stateI).get(stateJ))
				+ Math.log(emit.get(stateJ).get(observations[time]))+ getBeta(stateJ, time+1, observations)
				- getBeta(stateI, time, observations);
		return xi;
	}
	
	public HashMap<String, Double> updataPrior(ArrayList<String[]> trainingData){
		double priorProb;
		HashMap<String, Double> updatedPrior= new HashMap<String, Double>();
		for(String state: states){
			priorProb = 0;
			for(int i=0; i<trainingData.size(); i++){
				alphaRecord = new HashMap<Integer, HashMap<String, Double>>();
				betaRecord = new HashMap<Integer, HashMap<String, Double>>();
				if(i==0)
					priorProb = getGamma(state, 1, trainingData.get(i));
				else
					priorProb = Util.logSum(priorProb, getGamma(state, 1, trainingData.get(i)));
			}
			priorProb -= Math.log(trainingData.size());
			updatedPrior.put(state, Math.exp(priorProb));
		}
		return updatedPrior;
	}

	public HashMap<String, HashMap<String, Double>> updateTrans(ArrayList<String[]> trainingData){
		HashMap<String, HashMap<String, Double>> updatedTrans = new HashMap<String, HashMap<String, Double>>();
		double transProb1;
		double transProb2;
		for(String stateI: states){
			transProb2 = 0;
			HashMap<String, Double> probValue = new HashMap<String, Double>();
			for(int j=0; j<states.length; j++){
				transProb1 = 0;
				for(int m=0; m<trainingData.size();m++){
					alphaRecord = new HashMap<Integer, HashMap<String, Double>>();
					betaRecord = new HashMap<Integer, HashMap<String, Double>>();
					for(int t=1; t<trainingData.get(m).length; t++){
						if(t==1 && m==0)
							transProb1 = getXi(stateI, states[j], t, trainingData.get(m));
						else
							transProb1 = Util.logSum(transProb1, getXi(stateI, states[j], t, trainingData.get(m)));
					}
				}
				probValue.put(states[j], transProb1);
				if(j==0)
					transProb2 = transProb1;
				else
					transProb2 = Util.logSum(transProb2, transProb1);
			}
			for(String state: states){
				transProb1 = probValue.get(state);
				transProb1 -= transProb2;
				probValue.put(state, Math.exp(transProb1));
			}
			updatedTrans.put(stateI, probValue);
		}
		return updatedTrans;
	}

	public HashMap<String, HashMap<String, Double>> updateEmit(ArrayList<String[]> trainingData){
		double emitProb1;
		double emitProb2;
		int firstIndex = 0;
		HashMap<String, HashMap<String, Double>> updatedEmit = new  HashMap<String, HashMap<String, Double>>();
		for(String state: states){
			HashMap<String, Double> probValue = new HashMap<String, Double>();
			emitProb2 = 0;	
			for(int m=0; m<trainingData.size();m++){
				alphaRecord = new HashMap<Integer, HashMap<String, Double>>();
				betaRecord = new HashMap<Integer, HashMap<String, Double>>();
				for(int t=1; t<=trainingData.get(m).length; t++){
					if(t==1 && m==0){
						emitProb2 = getGamma(state, t, trainingData.get(m));
					}else{
						emitProb2 = Util.logSum(emitProb2, getGamma(state, t, trainingData.get(m)));
					}
				}
			}
			for(String word: vocabulary){
				emitProb1 = 0;
				firstIndex = 0;
				for(int m=0; m<trainingData.size();m++){
					alphaRecord = new HashMap<Integer, HashMap<String, Double>>();
					betaRecord = new HashMap<Integer, HashMap<String, Double>>();
					for(int t=1; t<=trainingData.get(m).length; t++){
						if(word.equals(trainingData.get(m)[t-1])){
							if(firstIndex == 0){
								firstIndex++;
								emitProb1 = getGamma(state, t, trainingData.get(m));
							}else{
								emitProb1 = Util.logSum(emitProb1, getGamma(state, t, trainingData.get(m)));
							}		
						}
					}
				}
				emitProb1 -= emitProb2;
				probValue.put(word, Math.exp(emitProb1));
			}
			updatedEmit.put(state, probValue);
		}
		return updatedEmit;
	}

	public double getAvgLL(ArrayList<String[]> trainingData){
		double avgLL;
		double sum = 0;
		for(String[] data: trainingData){
			alphaRecord = new HashMap<Integer, HashMap<String, Double>>();
			avgLL = 0;
			for(int i=0; i<states.length; i++){
				if(i==0)
					avgLL = getAlpha(states[i], data.length , data);
				else
					avgLL = Util.logSum(avgLL, getAlpha(states[i], data.length , data));
			}
			sum += avgLL;
		}
		return sum/trainingData.size();
	}

	public void init(String[] args){
		trainingData = new ArrayList<String[]>();
		vocabulary = new HashSet<String>();
		String train = args[0];
		File file = new File(train);
		String line;
		try {
			FileReader fr = new FileReader(file);
			BufferedReader br = new BufferedReader(fr);
			while((line=br.readLine())!=null){
				String[] tokens = line.split(" ");
				for(String token: tokens)
					vocabulary.add(token);
				trainingData.add(tokens);
			}	
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		if(args.length == 4){
			loadParameter(args[1], args[2], args[3]);
			return;
		}
		double sum = 0;
		double prob;
		//initialize prior
		for(String state: states){
			prob = 0.1+Math.random();
			prior.put(state, prob);
			sum += prob;
		}
		for(String state: states){
			prob = prior.get(state);
			prior.put(state, prob/sum);
		}
		//initialize trans
		for(String stateI: states){
			sum = 0;
			HashMap<String, Double> transValue = new HashMap<String, Double>();
			for(String stateJ: states){
				prob = 0.1+Math.random();
				transValue.put(stateJ, prob);
				sum += prob;
			}
			Iterator<String> it =  transValue.keySet().iterator();
			while(it.hasNext()){
				String key = it.next();
				prob = transValue.get(key);
				transValue.put(key, prob/sum);
			}
			trans.put(stateI, transValue);
		}
		//initialize emit
		for(String state: states){
			sum = 0;
			HashMap<String, Double> emitValue = new HashMap<String, Double>();
			for(String word: vocabulary){
				prob = 0.1+Math.random();
				emitValue.put(word, prob);
				sum += prob;
			}
			Iterator<String> it =  emitValue.keySet().iterator();
			while(it.hasNext()){
				String key = it.next();
				prob = emitValue.get(key);
				emitValue.put(key, prob/sum);
			}
			emit.put(state, emitValue);
		}
	}

	public void loadParameter(String transPath, String emitPath, String priorPath){
		String line;
		//load transfer probability
		try{
			File transf = new File(transPath);
			FileReader transFr = new FileReader(transf);
			BufferedReader transBr = new BufferedReader(transFr);
			while((line = transBr.readLine())!=null){
				HashMap<String, Double> transPro =  new HashMap<String, Double>();
				String[] seg = line.split(" ");
				for(int i =1; i <seg.length; i++){
					String[] value = seg[i].split(":");
					transPro.put(value[0], Double.parseDouble(value[1]));
				}
				trans.put(seg[0], transPro);	
			}
			transBr.close();

			//load emit probability
			File emitf = new File(emitPath);
			FileReader emitFr = new FileReader(emitf);
			BufferedReader emitBr = new BufferedReader(emitFr);
			while((line = emitBr.readLine())!=null){
				HashMap<String, Double> emitPro =  new HashMap<String, Double>();
				String[] seg = line.split(" ");
				for(int i =1; i <seg.length; i++){
					String[] value = seg[i].split(":");
					emitPro.put(value[0], Double.parseDouble(value[1]));
				}
				emit.put(seg[0], emitPro);	
			}
			emitBr.close();

			//load prior probability
			File priorf = new File(priorPath);
			FileReader priorFr = new FileReader(priorf);
			BufferedReader priorBr = new BufferedReader(priorFr);
			while((line = priorBr.readLine())!=null){
				String[] seg = line.split(" ");
				prior.put(seg[0], Double.parseDouble(seg[1]));	
			}
			priorBr.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
