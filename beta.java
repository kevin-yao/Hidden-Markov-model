import java.util.*;
import java.util.Map.Entry;
import java.io.*;
public class beta
{
	private HashMap<String, HashMap<String, Double>> trans;
	private HashMap<String, HashMap<String, Double>> emit;
	private static HashMap<Integer, HashMap<String, Double>> betaRecord;
	private HashMap<String, Double> prior;
	public beta(){
		trans = new HashMap<String, HashMap<String, Double>>();
		emit = new HashMap<String, HashMap<String, Double>>();
		prior = new HashMap<String, Double>();
	}
	public static void main(String[] args) throws IOException{
		beta be = new beta();
		be.evaluation(args);
	} 
	public void evaluation(String[] args) throws IOException{
		String devPath = args[0];
		String hmmTransPath = args[1];
		String hmmEmitPath = args[2];
		String hmmPriorPath = args[3];
		int middleTime = 1;
		loadParameter(hmmTransPath, hmmEmitPath, hmmPriorPath);
		File devf = new File(devPath);
		FileReader devFr = new FileReader(devf);
		BufferedReader devBr = new BufferedReader(devFr);
		String line;
		while((line = devBr.readLine())!=null){
			double likelihood1 = 0;
			String[] observations = line.split(" ");
			Iterator<String> it = prior.keySet().iterator();
			String state = it.next();
			betaRecord = new  HashMap<Integer, HashMap<String, Double>>();
			//sum all state
			alpha al = new alpha();
			al.loadParameter(hmmTransPath, hmmEmitPath, hmmPriorPath);
			likelihood1 = al.getAlpha(state, middleTime, line)+getBeta(state, middleTime , line);
			//System.out.println(al.getAlpha(state, 1, line));
			while(it.hasNext()){
				state = it.next();
				double likelihood2 = al.getAlpha(state, middleTime, line)+getBeta(state, middleTime, line);
				likelihood1 = Util.logSum(likelihood1, likelihood2);
			}
			System.out.println(likelihood1);
		}
		devBr.close();
	}
	//return log likelihood
	public double getBeta(String state, int time, String observation){
		double prob = 0;
		HashMap<String, Double>  betaValue;
		String[] observations = observation.split(" ");
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
			prob1 = getBeta(possibleState, time+1, observation)+Math.log(trans.get(state).get(possibleState))
					+ Math.log(emit.get(possibleState).get(observations[time]));
			while(it.hasNext()){
				entry = it.next();
				possibleState = entry.getKey();
				prob2 = getBeta(possibleState, time+1, observation)+Math.log(trans.get(state).get(possibleState))
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

	public void loadParameter(String transPath, String emitPath, String priorPath)throws IOException{
		String line;
		//load transfer probability
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
	}
}