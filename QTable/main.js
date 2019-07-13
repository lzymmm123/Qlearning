var learningRate = 0.1;
var gamma = 0.3;
var epsilon = 1;
var rows = 30;
var cols = 30;
var target;
var gap;
var actions = 4;
var Grids = [];
var states;
var table;
var player;
var iters = 500;
var maxstep = 1000;
var iter = 0;
var num = 0;
var startFlag = false;
class Grid{
    constructor(x,y,w,flag){
        this.x = x;
        this.y = y;
        this.w = w;
        this.flag = 0;
        if(arguments.length==4){
            this.flag = flag;
        }
    }
    addFlag(flag){
        this.flag = flag;
    }
    display(){
        fill(255);
        if(this.flag==0){
            fill(255);
        }
        else if(this.flag == 1){
            fill(255,0,0);
        }
        rect(this.x,this.y,this.w,this.w);
    }
}
class Player{
    constructor(i,j,radius){
        this.i = i;
        this.j = j;
        this.x = j*gap+radius;
        this.y = i*gap+radius;
        this.radius = radius;
    }
    display(){
        fill(0,255,0);
        this.x = this.j*gap+this.radius;
        this.y = this.i*gap+this.radius;
        ellipse(this.x,this.y,this.radius,this.radius);
    }
    move(dir){
        this.j += dir[0];
        this.i += dir[1];
        this.x = this.j*gap+this.radius;
        this.y = this.i*gap+this.radius;
    }
}
class Target{
    constructor(i,j,radius){
        this.i = i;
        this.j = j;
        this.x = j*gap+radius;
        this.y = i*gap+radius;
        this.radius = radius;
    }
    display(){
        fill(0,0,255);
        ellipse(this.x,this.y,this.radius,this.radius);
    }
}
function setup(){
    createCanvas(600,600);
    //frameRate(10);
    gap = floor(width/cols);
    states = rows*cols;
    Grids = new Array(states).fill().map((val,index) => {
        let i = floor(index/cols);
        let j = index%rows;
        return new Grid(j*gap,i*gap,gap);
    });
    initTable();
    
    player = new Player(5,5,gap/2);
    target = new Target(25,25,gap/2);
    addFlag(target.x,target.y,2);
    
}

function draw(){
    background(51);
    drawGrids();
    player.display();
    target.display();
    if(startFlag){
        start();
    }
    
}

function keyPressed(){
    if(key == "s"){
        startFlag = !startFlag;
    }
}
function start(){
    num++;
    if(num<maxstep&&(player.i != target.i||player.j!=target.j)){
        updateTable();
    }
    else{
        iter++;
        console.log(iter);
        num = 0;
        epsilon = 0.01 + (1 - 0.01)*pow(2.17,(-0.01*iter));
        //initTable();
        player = new Player(5,5,gap/2);
    }
 
}
function drawGrids(){
    for(let grid of Grids){
        grid.display();
    }
}
function offScreen(i,j){
    if(i<0||i>=rows||j<0||j>=cols){
        return true;
    }else{
        return false;
    }
}
function getReward(state){
    if (Grids[state].flag==0) {
        return -1;
    } else if(Grids[state].flag == 1){
        return -100;
    }else if(Grids[state].flag==2){
        return 100;
    }
}
function explore(){
    let r = floor(random()*4)%4;
    let curState = player.i*cols+player.j;
    let i = player.i;
    let j = player.j;

    let nextSate;
    if(r == 0){//up
        i -= 1;
    }else if(r==1){//down
        i += 1;
    }else if(r==2){//left
        j -= 1;
    }else{//right
        j += 1;
    }
    if(!offScreen(i,j)){
        player.i = i;
        player.j = j;
        nextSate = player.i*cols+player.j;
        //console.log(nextSate);
        let reward = getReward(nextSate);
        //console.log(max(table[nextSate]));
        table[curState][r] += learningRate*(reward+gamma*(max(table[nextSate])-table[curState][r])); 
    }
}
function exploit(){
    let i = player.i;
    let j = player.j;
    let curState = player.i*cols+player.j;
    let nextSate;
    let r = table[curState].indexOf(max(table[curState]));
    if(r == 0){//up
        i -= 1;
    }else if(r==1){//down
        i += 1;
    }else if(r==2){//left
        j -= 1;
    }else{//right
        j += 1;
    }
    if(!offScreen(i,j)){
        player.i = i;
        player.j = j;
        nextSate = player.i*cols+player.j;
        let reward =getReward(nextSate);
        //console.log(max(table[nextSate]));
        table[curState][r] += learningRate*(reward+gamma*(max(table[nextSate])-table[curState][r])); 
        //console.log(table[curState][r]);
    }
}
function updateTable(){
    let r = random();
    if(r<epsilon){
        explore();
    }else{
        exploit();
    }
}
function mousePressed() {
    if(mouseX>0&&mouseX<width&&mouseY>0&&mouseY<height){
        addFlag(mouseX,mouseY,1);
    }


}
function addFlag(x,y,flag){
    let i = floor(y/gap);
    let j = floor(x/gap);
    let index = i*cols+j;
    Grids[index].addFlag(flag);
}
function initTable() {
    table = new Array(states).fill().map(() => Array(actions).fill(0));
    for(let i = 0;i<cols;i++){
        table[i][0] = -Infinity;
        table[i+cols*(rows-1)][1] = -Infinity;
    }
    for(let i = 0;i<rows;i++){
        table[cols*i][2] = -Infinity;
        table[cols*i+(cols-1)][3] = -Infinity;
    }
}
