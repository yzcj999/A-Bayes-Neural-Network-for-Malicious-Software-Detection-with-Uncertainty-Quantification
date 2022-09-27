/*   */
 $(window).load(function(){
             $(".loading").fadeOut()
            })  
$(function () {
    echarts_1();
	// echarts_2();
	echarts_3();
	echarts_4();
	echarts_5();
	zb1();
	zb2();
	zb3();
    function echarts_1() {
        // 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('echart1'));
        option = {
                    tooltip : {
                        trigger: 'item',
                        formatter: "{b} : {c} ({d}%)"
                    },
                    legend: {
                        right:0,
                        top:90,
                        height:160,
                        itemWidth:8,
                        itemHeight:10,
                        itemGap:10,
                        textStyle:{
                            color: 'rgba(255,255,255,.6)',
                            fontSize:10
                        },
                        orient:'vertical',
                        data:['Allaple.A','Lolyda.AA 1','VB.AT','C2Lop.P','Yuner.A']
                    },
                   calculable : true,
                    series : [
                        {
                            name:' ',
							color: ['#62c98d', '#2f89cf', '#4cb9cf', '#53b666', '#62c98d', '#205acf', '#c9c862', '#c98b62', '#c962b9', '#7562c9','#c96262','#c25775','#00b7be'],	
                            type:'pie',
                            radius : [30, 70],
                            center : ['35%', '50%'],
                            roseType : 'radius',
                            label: {
                                normal: {
                                    show: true
                                },
                                emphasis: {
                                    show: true
                                }
                            },

                            lableLine: {
                                normal: {
                                    show: true
                                },
                                emphasis: {
                                    show: true
                                }
                            },

                            data:[
                                {value:10, name:'Allaple.A'},
                                {value:5, name:'Lolyda.AA 1'},
                                {value:15, name:'VB.AT'},
                                {value:25, name:'C2Lop.P'},
                                {value:20, name:'Yuner.A'},
                      
                            ]
                        },
                    ]
                };

        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }
    /*   */
// function echarts_2() {
//         // 基于准备好的dom，初始化echarts实例
//         var myChart = echarts.init(document.getElementById('echart2'));

//         option = {
//             tooltip: {
//                 trigger: 'item',
//                formatter: "{b} : {c} ({d}%)"
//             },
//             legend: {
			
// 				top:'15%',
//                 data: ['图例1', '图例2', '图例3', '图例4', '图例5'],
//                 icon: 'circle',
//                 textStyle: {
//                     color: 'rgba(255,255,255,.6)',
//                 }
//             },
//             calculable: true,
//             series: [{
//                 name: '',
// 				color: ['#62c98d', '#2f89cf', '#4cb9cf', '#53b666', '#62c98d', '#205acf', '#c9c862', '#c98b62', '#c962b9','#c96262'],	
//                 type: 'pie',
//                 //起始角度，支持范围[0, 360]
//                 startAngle: 0,
//                 //饼图的半径，数组的第一项是内半径，第二项是外半径
//                 radius: [51, 100],
//                 //支持设置成百分比，设置成百分比时第一项是相对于容器宽度，第二项是相对于容器高度
//                 center: ['50%', '45%'],
				
//                 //是否展示成南丁格尔图，通过半径区分数据大小。可选择两种模式：
//                 // 'radius' 面积展现数据的百分比，半径展现数据的大小。
//                 //  'area' 所有扇区面积相同，仅通过半径展现数据大小
//                 roseType: 'area',
//                 //是否启用防止标签重叠策略，默认开启，圆环图这个例子中需要强制所有标签放在中心位置，可以将该值设为 false。
//                 avoidLabelOverlap: false,
//                 label: {
//                     normal: {
//                         show: true,
//                       //  formatter: '{c}辆'
//                     },
//                     emphasis: {
//                         show: true
//                     }
//                 },
//                 labelLine: {
//                     normal: {
//                         show: true,
//                         length2: 1,
//                     },
//                     emphasis: {
//                         show: true
//                     }
//                 },
//                 data: [
//                     {value: 1,name: '0',},
//                     {value: 4,name: '1',},
//                     {value: 5,name: '2',},
//                     {value: 6,name: '3',},
//                     {value: 9,name: '4',},
         
                   

//                     {value: 0, name: "000",label: {show: false},labelLine: {show: false}},
//                     {value: 0, name: "111",label: {show: false},labelLine: {show: false}},
//                     {value: 0, name: "222",label: {show: false},labelLine: {show: false}},
//                     {value: 0, name: "333",label: {show: false},labelLine: {show: false}},
//                     {value: 0, name: "444",label: {show: false},labelLine: {show: false}},

                   
//                 ]
//             }]
//         };

//         // 使用刚指定的配置项和数据显示图表。
//         myChart.setOption(option);
//         window.addEventListener("resize",function(){
//             myChart.resize();
//         });
//     }
function echarts_3() {
        // 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('echart3'));

        option = {
	tooltip: {
		trigger: 'axis',
		axisPointer: {
			lineStyle: {
				color: '#57617B'
			}
		}
	},
	legend: {
	
		//icon: 'vertical',
			data: ['效果反馈', '用户查询量'],
        //align: 'center',
       // right: '35%',
		top:'0',
        textStyle: {
            color: "#fff"
        },
       // itemWidth: 15,
       // itemHeight: 15,
        itemGap: 20,
	},
	grid: {
		left: '0',
		right: '20',
		top:'10',
		bottom: '20',
		containLabel: true
	},
	xAxis: [{
		type: 'category',
		boundaryGap: false,
		axisLabel: {
			show: true,
			textStyle: {
                           color: 'rgba(255,255,255,.6)'
                        }
		},
		axisLine: {
			lineStyle: {
				color: 'rgba(255,255,255,.1)'
			}
		},
		data: ['4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月', '1月', '2月', '3月']
	}, {
		

		
		
	}],
	yAxis: [{
		axisLabel: {
			show: true,
			textStyle: {
                           color: 'rgba(255,255,255,.6)'
                        }
		},
		axisLine: {
			lineStyle: {
				color: 'rgba(255,255,255,.1)'
			}
		},
		splitLine: {
			lineStyle: {
				color: 'rgba(255,255,255,.1)'
			}
		}
	}],
	series: [{
		name: '效果反馈',
		type: 'line',
		smooth: true,
		symbol: 'circle',
		symbolSize: 5,
		showSymbol: false,
		lineStyle: {
			normal: {
				width: 2
			}
		},
		areaStyle: {
			normal: {
				color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
					offset: 0,
					color: 'rgba(24, 163, 64, 0.3)'
				}, {
					offset: 0.8,
					color: 'rgba(24, 163, 64, 0)'
				}], false),
				shadowColor: 'rgba(0, 0, 0, 0.1)',
				shadowBlur: 10
			}
		},
		itemStyle: {
			normal: {
				color: '#cdba00',
				borderColor: 'rgba(137,189,2,0.27)',
				borderWidth: 12
			}
		},
		data: [220, 182, 191, 134, 150, 120, 110, 125, 145, 122, 165, 122]
	}, {
		name: '用户查询量',
		type: 'line',
		smooth: true,
		symbol: 'circle',
		symbolSize: 5,
		showSymbol: false,
		lineStyle: {
			normal: {
				width: 2
			}
		},
		areaStyle: {
			normal: {
				color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
					offset: 0,
					color: 'rgba(39, 122,206, 0.3)'
				}, {
					offset: 0.8,
					color: 'rgba(39, 122,206, 0)'
				}], false),
				shadowColor: 'rgba(0, 0, 0, 0.1)',
				shadowBlur: 10
			}
		},
		itemStyle: {
			normal: {
				color: '#277ace',
				borderColor: 'rgba(0,136,212,0.2)',
				borderWidth: 12
			}
		},
		data: [120, 110, 125, 145, 122, 165, 122, 220, 182, 191, 153, 160]
	}]
};

        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }
function echarts_4() {
        // 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('echart4'));
option = {
   	tooltip: {
		trigger: 'axis',
		axisPointer: {
			lineStyle: {
				color: '#57617B'
			}
		}
	},
    "legend": {
		
      "data": [
        {"name": "事件数量"},
        {"name": "损失金额"},
        {"name": "比率"}
      ],
      "top": "0%",
      "textStyle": {
       "color": "rgba(255,255,255,0.9)"//图例文字
      }
    },
	
    "xAxis": [
      {
        "type": "category",
		
        data: ['4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月', '1月', '2月', '3月'],
		 axisLine: { lineStyle: {color: "rgba(255,255,255,.1)"}},
        axisLabel:  { textStyle: {color: "rgba(255,255,255,.6)", fontSize: '14', },
            },
		
        },
	],
    "yAxis": [
      {
        "type": "value",
        "name": "金额/万元",
        "min": 0,
        "max": 100,
        "interval": 20,
        "axisLabel": {
          "show": true,
         
        },
        axisLine: {lineStyle: {color: 'rgba(255,255,255,.4)'}},//左线色
        
      },
      {
        "type": "value",
        "name": "数量/个",
        "show": true,
        "axisLabel": {
          "show": true,
        
        },
		  axisLine: {lineStyle: {color: 'rgba(255,255,255,.4)'}},//右线色
		   splitLine: {show:true,lineStyle: {color:"#001e94"}},//x轴线
      },
    ],
    "grid": {
      "top": "10%",
		"right":"30",
		"bottom":"30",
		"left":"30",
    },
    "series": [
      {
        "name": "损失金额",
		  
        "type": "bar",
        "data": [82,35,79,80,42,73,84,93,98,76,38,52],
        "barWidth": "auto",
        "itemStyle": {
          "normal": {
            "color": {
              "type": "linear",
              "x": 0,
              "y": 0,
              "x2": 0,
              "y2": 1,
              "colorStops": [
                {
                  "offset": 0,
                  "color": "#609db8"
                },
                
                {
                  "offset": 1,
                  "color": "#609db8"
                }
              ],
              "globalCoord": false
            }
          }
        }
      },
      {
        "name": "事件数量",
        "type": "bar",
        "data": [
            68,25,64,51,29,69,54,70,83,66,18,12
        ],
        "barWidth": "auto",
		
        "itemStyle": {
          "normal": {
            "color": {
              "type": "linear",
              "x": 0,
              "y": 0,
              "x2": 0,
              "y2": 1,
              "colorStops": [
                {
                  "offset": 0,
                  "color": "#66b8a7"
                },
                {
                  "offset": 1,
                  "color": "#66b8a7"
                }
              ],
              "globalCoord": false
            }
          }
        },
        "barGap": "0"
      },
      {
        "name": "比率",
        "type": "line",
        "yAxisIndex": 1,
		
        "data": [65,50,48,70,53,40,64,33,51,28,64,82],
		  lineStyle: {
			normal: {
				width: 2
			},
		},
        "itemStyle": {
          "normal": {
            "color": "#cdba00",
			 
          }
        },
        "smooth": true
      }
    ]
};
       

        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }
	
function echarts_5() {
        // 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('echart5'));
// 颜色
var lightBlue = {
	type: 'linear',
	x: 0,
	y: 0,
	x2: 0,
	y2: 1,
	colorStops: [{
		offset: 0,
		color: 'rgba(41, 121, 255, 1)'
	}, {
		offset: 1,
		color: 'rgba(0, 192, 255, 1)'
	}],
	globalCoord: false
}

var option = {
	tooltip: {
		show: false
	},
	grid: {
		top: '0%',
		left: '65',
		right: '15%',
		bottom: '0%',
	},
	xAxis: {
		min: 0,
		max: 100,
		splitLine: {
			show: false
		},
		axisTick: {
			show: false
		},
		axisLine: {
			show: false
		},
		axisLabel: {
			show: false
		}
	},
	yAxis: {
		data: ['加密矿工', '变种组合','后门程序','间谍程序','木马','蠕虫','病毒','勒索软件','僵尸网络','广告程序'],
		//offset: 15,
		axisTick: {
			show: false
		},
		axisLine: {
			show: false
		},
		axisLabel: {
			color: 'rgba(255,255,255,.6)',
			fontSize: 14
		}
	},
	series: [{
		type: 'bar',
		label: {
			show: true,
			zlevel: 10000,
			position: 'right',
			padding: 8,
			color: '#49bcf7',
			fontSize: 13,
			formatter: '{c}%'
			
		},
		itemStyle: {
			color:'#49bcf7'
		},
		barWidth: '15',
		data: [80, 67, 99, 12, 19, 39, 84, 28, 47, 57, 100],
		z: 5
	}, {
		type: 'bar',
		barGap: '-100%',
		itemStyle: {
			color:'#fff',
			opacity: 0.1
		},
		barWidth: '15',
		data: [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
		z: 3
	}],
};
        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }
	
	
function zb1() {
        // 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('zb1'));
	var v1=421//正常程序
		var v2=612//恶意程序
		var v3=v1+v2//总检测量 
option = {	
    tooltip : {
        trigger: 'item',
        formatter: "{b} : {c} ({d}%)"
    },
   calculable : true,
    series: [{
		
        type: 'pie',
        radius: ['60%', '70%'],
        color:'#49bcf7',
        label: {
            normal: {
                position: 'center'
            }
        },
        data: [{
            value: v2,
            name: '恶意程序',
            label: {
                normal: {
                    formatter: v2 +'',
                    textStyle: {
                        fontSize: 20,
						color:'#fff',
                    }
                }
            }
        }, {
            value: v1,
            name: '正常程序',
            label: {
                normal: {
                 formatter : function (params){
                return '占比'+Math.round( v2/v3*100)+ '%'
            },
                    textStyle: {
                        color: '#aaa',
                        fontSize: 12
                    }
                }
            },
            itemStyle: {
                normal: {
                    color: 'rgba(255,255,255,.2)'
                },
                emphasis: {
                    color: '#fff'
                }
            },
        }]
    }]
};
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }
function zb2() {
        // 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('zb2'));
 var v1=298//恶意程序
		var v2=523//正常程序
		var v3=v1+v2//总检测量 
      option = {
        tooltip : {
            trigger: 'item',
            formatter: "{b} : {c} ({d}%)"
        },
       calculable : true,
	
//animation: false,
    series: [{	
        type: 'pie',
       radius: ['60%', '70%'],
        color:'#cdba00',
        label: {
            normal: {
                position: 'center'
            }
        },
        data: [{
            value: v1,
            name: '恶意程序',
            label: {
                normal: {
                    formatter: v1 +'',
                    textStyle: {
                        fontSize: 20,
						color:'#fff',
                    }
                }
            }
        }, {
            value: v2,
            name: '正常程序',
            label: {
                normal: {
                 formatter : function (params){
                return '占比'+Math.round( v1/v3*100)+ '%'
            },
                    textStyle: {
                        color: '#aaa',
                        fontSize: 12
                    }
                }
            },
            itemStyle: {
                normal: {
                    color: 'rgba(255,255,255,.2)'
                },
                emphasis: {
                    color: '#fff'
                }
            },
        }]
    }]
};
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }
/*   */
function zb3() {
        // 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('zb3'));
		var v1=298//正常程序
		var v2=523//恶意程序
		var v3=v1+v2//总检测量 
option = {	
    tooltip : {
        trigger: 'item',
        formatter: "{b} : {c} ({d}%)"
    },
   calculable : true,
    series: [{
		
        type: 'pie',
       radius: ['60%', '70%'],
        color:'#62c98d',
        label: {
            normal: {
                position: 'center'
            }
        },
        data: [{
            value: v2,
            name: '恶意程序',
            label: {
                normal: {
                    formatter: v2 +'',
                    textStyle: {
                        fontSize: 20,
						color:'#fff',
                    }
                }
            }
        }, {
            value: v1,
            name: '正常程序',
            label: {
                normal: {
                 formatter : function (params){
                return '占比'+Math.round( v2/v3*100)+ '%'
            },
                    textStyle: {
                        color: '#aaa',
                        fontSize: 12
                    }
                }
            },
            itemStyle: {
                normal: {
                    color: 'rgba(255,255,255,.2)'
                },
                emphasis: {
                    color: '#fff'
                }
            },
        }]
    }]
};
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }
})



		
		
		


		









