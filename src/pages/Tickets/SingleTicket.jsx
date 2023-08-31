import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPen } from '@fortawesome/free-solid-svg-icons';
import bg from '../../images/mainbg.jpg';
import ai from '../../images/ai.gif';
import Sidepanel from "../../components/sidepanel";


function SingleTicket() {

   
return(

  <div className="main-body h-screen w-full ">
    <div className="main-body-container w-full flex flex-row absolute">
      <Sidepanel />
      <div className="w-5/6 side-panel p-5 md:ml-[300px] ml-16">
   

    <div className="flex flex-row w-full overflow-y-auto"> 

        {/* left side of Right part*/}
      <div className="flex flex-col main-body-container w-4/6 p-5 lg:overflow-x-scroll  overflow-y-auto">  


        {/* div 1 start */}
        <div className="w-[100%] p-8 shadow-lg border bg-white border-slate-100 rounded-lg">
                {/* <span
                        className="relative inset-x-0 bottom-0 left-0  bottom-0 h-2 bg-gradient-to-r from-green-300 via-blue-500 to-purple-600 border border-gray-900"
                    >  
                </span> */}

                <div className="justify-between sm:flex">
                    <div>
                    <h5 className="text-xl font-bold text-slate-900">
                        Ticket ID - #TC54 
                    </h5>
                            {/* <p className="mt-1 text-xs font-medium text-slate-600">By Ana Doe</p> */}
                    </div>

                    <button className="group relative px-5 py-2 overflow-hidden rounded-2xl bg-green-500 text-sm font-bold text-white">
                    Assign to me
                        <div className="absolute inset-0 h-full w-full scale-0 rounded-2xl transition-all duration-300 group-hover:scale-100 group-hover:bg-white/30">

                        </div>
                    </button>
                </div>
            
                <div className="grid grid-cols-6 w-[80%] mt-2">
                    <div className="col-span-3 p-3">
                        <p className="text-gray-700 ">Current Date :<span className="font-semibold"> 2023/05/06</span></p>
                    </div>
                  
                     <div className=" col-span-3 p-3">
                        <p className="text-gray-700 ">Update Date :<span className="font-semibold"> 2023/05/07</span></p>  
                    </div>   

                    <div className=" col-span-3 p-3">
                        <p className="text-gray-700 ">Current Time :<span className="font-semibold"> 8.15 am</span></p>
                    </div>
                  
                    <div className=" col-span-3 p-3">
                        <p className="text-gray-700">Status :<span className="font-semibold"> Open</span></p>  
                    </div> 

                    <div className=" col-span-6 p-3">
                        <p className="text-gray-700 ">Description : </p>  
                    </div> 
                </div>  

                <div className="border shadow-md text-gray-700 border-gray-400 w-full h-auto bg-gray-100 p-2 rounded-lg">
                    <p>
                    Description comes here
                    </p>              
                </div>
            
                <div className="grid grid-cols-6 w-[100%] mt-2">
                    <div className="col-span-6 p-3">
                        <p className="text-gray-700 ">Progress</p>
                    </div>
                    <div className="col-span-6">
                         {/* <progress id="file" value="32" max="100" className="w-[100%]"> 32% </progress> */}

                        <div className="relative w-full bg-gray-300 rounded overflow-hidden">
                            <div id="progress" className="bg-red-400 h-2" style={{width:'10%'}}>

                            </div>
                            {/* to make this progressbar dynamic, write function to change width of the #progressbar */}
                        </div>
                    </div>   
                </div>
        </div> 
        {/* div 1 end */}



        {/* div 2 start */}
        <div className="w-[100%] mt-5 p-8 overflow shadow-lg border bg-white border-slate-100 rounded-lg">            
          <div>
            <div className="flex">
              <div className="mx-auto w-full">
                <form action="" method="POST">
                      
                  <div className="mb-5">
                    <textarea
                      rows="4"
                      name="message"
                      id="message"
                      placeholder="Type your message"
                      className="w-full resize-none rounded-md border border-[#e0e0e0] bg-white py-3 px-6 text-base font-medium text-[#6B7280] outline-none focus:border-[#6A64F1] focus:shadow-md"
                    ></textarea>
                  </div>
                  <div>
                    <button
                      className="hover:shadow-form rounded-md bg-[#6A64F1] py-3 px-8 text-base font-semibold text-white outline-none"
                    >
                      Submit
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>

        </div>
        {/* div 2 end */}


        {/* div 2.1 start */}
        <div classNameName="p-6 grid grid-cols-1 border border-blue-500">
          <div className="flex-1 bg-white rounded-lg shadow-xl mt-4 p-8 ">
            <h4 className="text-xl text-gray-900 font-bold">Time Line</h4>
            <div className="relative px-4">
              <div className="absolute h-full border border-dashed border-opacity-20 border-secondary">

              </div>

              <div className="flex items-center w-full my-6 -ml-1.5">
                <div className="w-1/12 z-10">
                    <div className="w-3.5 h-3.5 bg-blue-600 rounded-full">

                    </div>
                </div>
                <div className="w-11/12">
                    <p className="text-sm">Profile informations changed.</p>
                    <p className="text-xs text-gray-500">3 min ago</p>
                </div>
              </div>
                      
              <div className="flex items-center w-full my-6 -ml-1.5">
                <div className="w-1/12 z-10">
                  <div className="w-3.5 h-3.5 bg-blue-600 rounded-full">

                  </div>
                </div>
                <div className="w-11/12">
                  <p className="text-sm">
                    Connected with <a href="#" className="text-blue-600 font-bold">Colby Covington</a>.</p>
                  <p className="text-xs text-gray-500">15 min ago</p>
                </div>
              </div>
                      
              <div className="flex items-center w-full my-6 -ml-1.5">
                <div className="w-1/12 z-10">
                  <div className="w-3.5 h-3.5 bg-blue-600 rounded-full">

                  </div>
                </div>
                <div className="w-11/12">
                  <p className="text-sm">Invoice <a href="#" className="text-blue-600 font-bold">#4563</a> was created.</p>
                  <p className="text-xs text-gray-500">57 min ago</p>
                </div>
              </div>
                      
              <div className="flex items-center w-full my-6 -ml-1.5">
                <div className="w-1/12 z-10">
                  <div className="w-3.5 h-3.5 bg-blue-600 rounded-full">

                  </div>
                </div>
                <div className="w-11/12">
                  <p className="text-sm">
                      Message received from <a href="#" className="text-blue-600 font-bold">Cecilia Hendric</a>.</p>
                  <p className="text-xs text-gray-500">1 hour ago</p>
                </div>
              </div>
                      
              <div className="flex items-center w-full my-6 -ml-1.5">
                <div className="w-1/12 z-10">
                  <div className="w-3.5 h-3.5 bg-blue-600 rounded-full">

                  </div>
                </div>
                <div className="w-11/12">
                  <p className="text-sm">New order received <a href="#" className="text-blue-600 font-bold">#OR9653</a>.</p>
                  <p className="text-xs text-gray-500">2 hours ago</p>
                </div>
              </div>

              <div className="flex items-center w-full my-6 -ml-1.5">
                <div className="w-1/12 z-10">
                  <div className="w-3.5 h-3.5 bg-blue-600 rounded-full">

                  </div>
                </div>
                <div className="w-11/12">
                  <p className="text-sm">
                    Message received from <a href="#" className="text-blue-600 font-bold">Jane Stillman</a>.</p>
                  <p className="text-xs text-gray-500">2 hours ago</p>
                </div>
              </div>
                      
          </div>
        </div></div>
        {/* div 2.1 end */}
      
      </div>

      <div className="flex flex-col main-body-container w-[30%] h-full p-4">  

        {/* div 3 start */}
        <div className="flex w-full h-50 justify-center items-center mt-1">
          <div className="flex flex-row w-[100%] h-40 border border-red-400 bg-red-100 rounded-lg">
               <div className="w-[90%] p-5 md:text-xs lg:text-sm xl:text-base">
                <p>Import information</p>
               </div>
               <div className="justify-right items-right">
                <FontAwesomeIcon icon={faPen} className="md:text-xs lg:text-sm xl:text-base p-5"/>
               </div>
                
          </div>
        </div> 
        {/* div 3 end */}


        {/* div 4 start */}
        <div className="flex flex-col rounded-lg w-full h-80 mt-3 border ">
          <div className="w-full overflow-x-hidden sm:text-xs md:text-xs lg:text-xs xl:text-base rounded-t-lg bg-gray-700 text-white h-10 text-center pt-1 border border-gray-900">
            Manual - Ticket Support Team
          </div>

          <div className="flex flex-row w-full h-auto border p-3 overflow-x-auto ">
            <div className="">
              <img className="w-14 h-14 rounded-full border object-cover" src={bg}/>
            </div>
            <div className="w-3/5 p-5 md:text-xs lg:text-sm xl:text-sm flex flex-row">
                <p>dilakshilamahewa@gmail.com </p><FontAwesomeIcon icon={faPen} className="ml-1"/>
            </div>
          </div> 
          
          <div className="p-5 md:text-xs lg:text-xs xl:text-sm grid grid-cols-3 overflow-x-auto text-gray-600">
            <div className=" mt-2">Email -</div>
              <div className="col-span-2 mt-2">dilakshilamahewa@gmail.com</div>
            <div className=" mt-2">Post -</div>
              <div className="col-span-2 mt-2">Support Engineer</div>
            <div className=" mt-2">Phone -</div>
              <div className="col-span-2 mt-2"> 0704003184</div>
            <div  className="col-span-3 flex justify-end pr-5"><button className="bg-blue-600 px-4 py-1 rounded-lg text-gray-100 mt-5">Primary</button></div>

          </div>    
        </div> 
        {/* div 4 end */}


        {/* div 5 start */}
        <div className="flex flex-col rounded-lg w-full h-80 mt-3 border">
          <div className="w-[100%] md:text-xs lg:text-sm xl:text-base rounded-t-lg bg-gray-700 text-white h-10 text-center pt-1 border border-gray-900">
           Auto - AI Model
          </div>
          <div className="">
            <h1 className="text-xl font-bold text-center pt-3">Hello!</h1>
          </div>
          <div className=" flex  justify-center items-center mt-2">
           <img src={ai} className="w-1/2 h-auto "/>
          </div>
          <div className="md:text-xs lg:text-sm xl:text-base flex justify-center items-center mt-2">
           <p>How can I help you?</p>
          </div>
        </div>
        {/* div 5 end */}
      
      </div>
      </div>
    </div>
  </div>
  </div>

  


  );
}


export default SingleTicket;