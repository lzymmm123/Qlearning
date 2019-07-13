var rows;
var nn;
var grid;
var gap;
var grids=[];
var player;
var flag;
var enableConfig = true;
var cols;
var env;
var dqn;
class NN {
    constructor(input_shape,hidden_shape_array,output_shape) {
        this.input_shape = input_shape;
        this.hidden_shape_array = hidden_shape_array;
        this.output_shape = output_shape;
        this.model = this.createModel(input_shape,hidden_shape_array,output_shape);
        //this.model.compile({loss: 'meanSquaredError', optimizer: 'adam'});
    }
    copy(){
        //return tf.tidy(() => {
            const modelCopy = new NN(this.input_shape,this.hidden_shape_array,this.output_shape);
            const weights = this.model.getWeights();
            //console.log(weights);
            const weightCopies = [];
            for(let i = 0 ; i<weights.length;i++){
                weightCopies[i] = weights[i].clone();
            }
            modelCopy.model.setWeights(weightCopies);
            return modelCopy;
        //})
    }
    configOptimizer(learningRate,optimizerName){
        if(optimizerName=="sgd"){
            this.optimizer = tf.train.sgd(learningRate);
        }
        if(optimizerName=="adam"){
            this.optimizer = tf.train.adam(learningRate);
        }
    }
    loss(predict_ys,labels) {
        return tf.tidy(()=>{
            const meanSquareError = tf.losses.meanSquaredError(labels,predict_ys);//predict_ys.sub(labels).square().mean();
            const lossCross = tf.losses.sigmoidCrossEntropy(labels,predict_ys);
            return meanSquareError;
        });

    }
    train_step(xs,ys){
        let meanloss = 0;
        this.optimizer.minimize(()=>{
            let predict_ys = this.predict(xs);
            let l = this.loss(predict_ys,ys).mean();
            meanloss=l.dataSync();
            return l;
        })
        return meanloss;
        // return await this.model.fit(xs, ys,{
        //     shuffle:false
        // });
        // console.log("Loss after Epoch " + " : " + h.history.loss[0]);
    }
    createModel(input_shape,hidden_shape_array,output_shape){
        let model = tf.sequential();
        if(hidden_shape_array instanceof Array){
            for(let i = 0;i<hidden_shape_array.length;i++){
                if(i == 0){
                    model.add(tf.layers.dense({
                        units:hidden_shape_array[i],
                        inputShape:input_shape,
                        activation:"relu",
                    }));
                }else{
                    model.add(tf.layers.dense({
                        units:hidden_shape_array[i],
                        activation:"relu",
                    }));
                }
            }
        }
        else{
            model.add(tf.layers.dense({
                units:hidden_shape_array,
                inputShape:input_shape,
                activation:"relu",
            }))
            
        }
        model.add(tf.layers.dense({
            units:output_shape,
            activation:"relu",
        }));
        return model;
    }
    predict(x){
        if(x instanceof tf.Tensor){
            return tf.tidy(()=>{
                const output = this.model.predict(x);
                //const data = output.dataSync();
                return output;
            })
        }
        if(x instanceof Array){
            if(x[0] instanceof Array && x[0].length==this.input_shape){
                return tf.tidy(()=>{
                    const input = tf.tensor2d(x);
                    const output = this.model.predict(input);
                    return output;
                    // const data = output.dataSync();
                    // return data;
                });
            }
            else{
                return tf.tidy(()=>{
                    const input = tf.tensor2d([x]);
                    const output = this.model.predict(input);
                    return output;
                });
            }
        }
    }
}

class Ceiling {
    constructor(x,y,len,hei) {
        this.x = x;
        this.y = y;
        this.len = len;
        this.hei = hei;
    }
    display(){
        fill(255,0,0);
        rect(this.x,this.y,this.hei,this.len);
    }
    move(speed){
        this.x+=speed;
    }
}

class Grid {
    constructor(xOrGrid,y,w,flag=0) {
        if(xOrGrid instanceof Grid){
            this.x = xOrGrid.x;
            this.y = xOrGrid.y;
            this.w = xOrGrid.w;
            this.flag = xOrGrid.flag;
            this.score = xOrGrid.score;
        }else{
            this.x = xOrGrid;
            this.y = y;
            this.w = w;
            this.flag = flag;
            this.score = -0.05;
        }
    }
    copy(){
        return new Grid(this);
    }
    display(){
        stroke(0);
        strokeWeight(1);
        switch (this.flag) {
            case 0:
                fill(255,255,255);//道路
                break;
            case 1:
                fill(255,0,0);//毒药 
                this.score = -0.75;
                break;
            case 2:
                fill(0,255,0);//基础奖励
                this.score = 0.1;
                break;
            case 3:
                fill(0,0,255);//最终目标
                this.score = 1;
                break;
            default:
                fill(255,255,255);
                break;
        }
        rect(this.x,this.y,this.w,this.w);
    }
    setFlag(flag){
        this.flag = flag;
    }
}
class Player {
    constructor(rowOrPlayer,col,radius,gap) {
        if(rowOrPlayer instanceof Player){
            this.row = rowOrPlayer.row;
            this.col =rowOrPlayer.col;
            this.gap =rowOrPlayer.gap;
            this.x =rowOrPlayer.x;
            this.y =rowOrPlayer.y;
            this.radius =rowOrPlayer.radius;
        }
        else{
            this.row = rowOrPlayer;
            this.col = col;
            this.gap = gap;
            this.x = col*gap+gap/2;
            this.y = rowOrPlayer*gap+gap/2;
            this.radius = radius;
            this.score = 0;
        }
    }
    copy(){
        return new Player(this);
    }
    display(){
        fill(0,0,0);
        ellipse(this.x,this.y,this.radius,this.radius);
    }
    resetXY(){
        this.x = this.col*this.gap+this.gap/2;
        this.y = this.row*this.gap+this.gap/2;
    }
    move(dir){//0 up  1 right  2 down 3 left  
        switch (dir) {
            case 0:
                if(this.row>0){
                    this.row -= 1;
                    this.resetXY();
                }
                break;
            case 1:
                if(this.col<cols-1){
                    this.col += 1;
                    this.resetXY();
                }
                break;
            case 2:
                if(this.row<rows-1){
                    this.row += 1;
                    this.resetXY();
                }
                break;
            case 3:
                if(this.col>0){
                    this.col -= 1;
                    this.resetXY();
                }
                break;
            default:
                console.log("error454654");
                break;
        }
    }
}
class DQN {
    constructor(input_shape,hidden_shape_array,output_shape,eplison,CSteps) {
        this.action_nn = new NN(input_shape,hidden_shape_array,output_shape);
        this.target_nn = this.action_nn.copy();
        this.eplison = eplison;
        this.CSteps = CSteps;
        this.memory = [];
        this.iter = 0;
    }
    setNNConfig(learningRate,optimizerName){
        this.action_nn.configOptimizer(learningRate,optimizerName);
    }
    setCSteps(CSteps){
        this.CSteps = CSteps;
    }
    setEplison(min_epliosn,max_eplison,init_eplison){
        this.min_epliosn = min_epliosn;
        this.max_eplison = max_eplison;
        this.eplison = init_eplison;
    }
    setMemorySize(num){
        this.memorySize = num;
    }
    pushMemory(example){
        if(this.memory.length<this.memorySize){
            this.memory.push(example);
        }else{
            this.memory.shift();
            this.memory.push(example);
        }
    }
    train_step(examples){
        this.iter++;
        let curState = [];
        let nextState = [];
        let actions = [];
        let rewards = [];
        let dones = [];
        let valids = [];
        for(let example of examples){
            curState.push(example.curState);
            nextState.push(example.nextState);
            let temAction = new Array(4).fill(0);
            temAction[example.action] = 1;
            actions.push(temAction);
            // let temReward = new Array(4).fill(0);
            // temReward[example.action] = example.reward;
            rewards.push(example.reward);
            if(example.done){
                dones.push(0);
            }else{
                dones.push(1);
            }
            if(example.valid){
               valids.push(1); 
            }else{
                valids.push(0);
            }
        }
        let meanloss;
        
        this.action_nn.optimizer.minimize(()=>{
            let curTensor = tf.tensor2d(curState);
            let nextTensor = tf.tensor2d(nextState);
            let actionsTensor = tf.tensor2d(actions);
            let rewardTensor = tf.tensor1d(rewards);
            let donesTensor = tf.tensor1d(dones);
            let validsTensor = tf.tensor1d(valids);
            let Qvalue = this.action_nn.predict(curTensor).mul(actionsTensor).sum(1);
            let targetValue = this.target_nn.predict(nextTensor);
            let tar = targetValue.max(1).mul(0.998).mul(donesTensor).mul(validsTensor).add(rewardTensor);
            let l = tf.square(Qvalue.sub(tar)).mean(); 
            meanloss = l.dataSync();
            return l;
        })
        return meanloss;
    }
    reduceEpsilon(iter){
        this.eplison = this.min_epliosn + (1 - this.min_epliosn)*pow(2.17,(-0.01*iter));
    }
    getAction(curState){
        let r = Math.random();
        let action;
        if(r>this.eplison){
            let a = this.action_nn.predict(curState);
            //action = a.indexOf(Math.min.apply(Math,a));
            action = a.argMax(1).dataSync()[0];
            if(action!=0&&action!=1&&action!=2&&action!=3){
                alert(a);
                alert(action);
            }
            //console.log(a instanceof Float32Array);
            // console.log("a  :  "+a);
            // console.log("action_choice:  "+action);
        }else{
            let a = [0,1,2,3];
            let row = curState[curState.length-2];
            let col = curState[curState.length-1];
            let index;
            if(row===0){
                index = a.indexOf(0);
                a.splice(index,1);
            }else if(row === rows-1){
                index = a.indexOf(2);
                a.splice(index,1);
            }
            if(col === 0){
                index = a.indexOf(3);
                a.splice(index,1);
            }else if(col === cols-1){
                index = a.indexOf(1);
                a.splice(index,1);
            }
            action =a[Math.floor(Math.random()*a.length)%a.length];
        }
        return action;
    }
    setTargetNN(){
        tf.tidy(()=>{
            this.target_nn = this.action_nn.copy();
        })
    }
}
class ENV {
    constructor(player,grids) {
        this.initPlayer = player.copy();
        this.initGrids = grids;
        // for(let grid of grids){
        //     this.initGrids.push(grid.copy());
        // }
        this.player = player;
        this.grids = grids;
        this.state = [];
        for(let grid of grids){
            this.state.push(grid.flag);
        }
        this.state.push(player.row);
        this.state.push(player.col);
    }
    action(dir){
        let new_state = this.state.slice();
        let cur_state = this.state.slice();
        let curRow = this.player.row;
        let curCol = this.player.col;
        this.player.move(dir);
        let index = this.player.row*cols+this.player.col;
        let reward = this.grids[index].score;
        let done = this.grids[index].flag==3;
        let valid = true;
        if(curRow==0&&dir==0){
            reward = -0.75;
            valid = false;
        }else if(curRow==rows-1&&dir==2){
            reward = -0.75;
            valid = false;
        }else if(curCol==0&&dir==3){
            reward = -0.75;
            valid = false;
        }else if(curCol==cols-1&&dir==1){
            reward = -0.75;
            valid = false;
        }
        new_state.pop();
        new_state.pop();
        new_state.push(this.player.row);
        new_state.push(this.player.col);
        this.state = new_state;
        return {curState:cur_state,action:dir,reward:reward,nextState:new_state,done:done,valid:valid};
        
    }
    reset(){
        this.player = this.initPlayer.copy();
        this.grids = this.initGrids.slice();   
    }
    display(){
        for(let grid of this.grids){
            grid.display();
        }
        this.player.display();
    }
}
$(document).ready(()=>{
    var poison = $("#poison");
    var reward = $("#reward");
    var target = $("#target");
    var commit = $("#commit");
    poison[0].onclick = function(){
        flag = 1;
        console.log(flag)
    }
    reward[0].onclick = function(){
        flag = 2;
    }
    target[0].onclick =  function(){
        flag = 3;
    }
    commit[0].onclick = function(){
        poison.remove();
        reward.remove();
        target.remove();
        flag = 0;
        enableConfig =false;
        $(this).text("重新设置");
    }
});
// var ceiling;
// var flooring;
function generateData(shape,weights) {
    return tf.tidy(()=>{
        const xs = tf.randomUniform([20,shape]);
        const weight_tensor = tf.tensor1d(weights);
        const ys = xs.mul(weight_tensor).sum(1).reshape([20,1]);
        return {xs,ys};
    })
}
function gridsDisplay(){
    for(let grid of grids){
        grid.display();
    }
}

var iter = 0;
var startFlag = false;
var maxstep = 100;
var step = 0;
function setup() {
    createCanvas(600,600);
    gap = 100;
    rows = floor(height/gap);
    cols = floor(width/gap);
    grid = new Grid(0,0,gap);
    for(let i = 0 ;i<rows*cols;i++){
        grids.push(new Grid((i%cols)*gap,floor(i/cols)*gap,gap));
    }
    player = new Player(4,4,gap/2,gap);
    env = new ENV(player,grids);
    var inputShape = cols*rows+2;
    // action_nn = new NN(inputShape,[inputShape+50,floor(inputShape/2)+1],4);
    // target_nn = action_nn.copy();
    dqn = new DQN(inputShape,[inputShape+50,floor(inputShape/2)+1],4,1,100);
    dqn.setCSteps(15);
    dqn.setMemorySize(1024);
    dqn.setEplison(0.01,1,1);
    dqn.setNNConfig(0.001,"adam");
    //nn.configOptimizer(0.001,"adam");
    // ceiling = new Ceiling(0,0,20,100);
    // flooring = new Ceiling(0,height-20,20,100);
}

// var movingRight = false;
// var movingLeft = false;
// var speed = 5;

function draw(){
    background(255);
    env.display();
    if(startFlag){
        tf.tidy(()=>{
            step++;
            let a = dqn.getAction(env.state);
            let example = env.action(a);
            dqn.pushMemory(example);
            if(dqn.memory.length>=dqn.memorySize){
                let loss = dqn.train_step(dqn.memory)
                console.log("iter "+iter+"  step  "+step+"  loss: "+loss);
            }
            if(example.done||step>=maxstep){
                env.reset();
                step = 0;
                iter++;
                dqn.reduceEpsilon(iter);
                if((iter+1)%dqn.CSteps==0){
                    dqn.setTargetNN();
                    console.log("q-->target");
                }
            }
        })
    }
    // gridsDisplay();
    // player.display();
    // ceiling.move(1);
    // ceiling.display();
    // flooring.display();
    // tf.tidy(()=>{
    //     let t0 = generateData(3,[2,3,4]);
    //     let l = nn.train_step(t0["xs"],t0["ys"])
    //     console.log(l);
    //     // let l = nn.train_step(xs,ys);
    //     // console.log(l);
    // })
    
}
async function a(params) {
    
    let t = {"xs":[[0,0],[0,1],[1,0],[1,1]],"ys":[[0],[1],[1],[0]]};
    
    //tf.tidy(()=>{
        const xs = tf.tensor2d(t["xs"]);
        const ys = tf.tensor2d(t["ys"]);
        // let t = generateData(3,[2,3,4]);
            //console.log("ys  "+t["ys"].dataSync());
       // let losst = 
        await nn.train_step(xs,ys);
        // console.log("  t_loss:"+losst);   
    //})
}
function keyPressed(){
    // if(key=="a"||key=="A"){
    //     movingLeft = true;
    // }
    // if(key=="d"||key=="D"){
    //     movingRight = true;
    // }
    if(key=="s"||key=="S"){
        startFlag = !startFlag;
        console.log("start");
    }
    return false;
}
// function keyReleased(){
//     if(key=="a"||key=="A"){
//         movingLeft = false;
//     }
//     if(key=="d"||key=="D"){
//         movingRight = false;
//     }

//     return false;
// }
function mousePressed(){
    if(mouseX>0&&mouseX<width&&mouseY>0&&mouseY<height){
        if(enableConfig){
            var index = floor(mouseY/gap)*cols+floor(mouseX/gap);
            grids[index].setFlag(flag);
        }
    }
}
