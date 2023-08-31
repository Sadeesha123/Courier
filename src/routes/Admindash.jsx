import {useState} from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faStarHalfStroke } from '@fortawesome/free-solid-svg-icons'
import Sidepanel from "../components/sidepanel";
import MonthlyProgress from "../components/chart";
import bg from '../images/bg2.jpg';
import Calendar from 'moedim';


function Dashboard()
{
    const [value, setValue] = useState(new Date())
    const [selectedImage, setSelectedImage] = useState(null);

    const handleImageChange = (event) => {
      const file = event.target.files[0];
      setSelectedImage(file);
    };
   

    return(

            <div className="main-body h-screen w-full bg-slate-100">
              <img src={bg} alt="" srcset="" className="object-cover w-[100%] h-[100%] bg-no-repeat bg-fixed fixed" />
                    <div className="main-body-container lg:h-screen w-full flex flex-row top-0 absolute bg-black bg-opacity-50">
                        <Sidepanel/>
                        <div className="w-5/6 side-panel  p-5 md:ml-[300px] ml-12">
                          <div className="common-body p-5 flex flex-col h-full ">

                            <div className="indicators w-[80%] flex lg:flex-row mx-auto justify-between flex-col">
                                
                                <div className=" p-3 lg:w-[33%] justify-center text-center rounded-lg flex flex-col shadow-md shadow-slate-400">
                                      <div className="bg-white h-full rounded-lg items-center justify-center flex flex-col">
                                      <span className="font-semibold text-red-800 text-[18px]">Delivered Packages</span>
                                      <div className="mt-3 flex flex-row justify-center items-center">
                                        <span className="text-red-800 text-[22px] mr-3">
                                          <FontAwesomeIcon icon={faPlus} />
                                        </span>
                                        <span className="text-red-800 text-[36px] font-bold">106</span>
                                      </div>
                                      </div>
                                      
                                </div>


                                <div className=" p-3 lg:w-[33%] justify-center text-center rounded-lg flex flex-col shadow-md shadow-slate-400">
                                      <div className="bg-white h-full rounded-lg items-center justify-center flex flex-col">
                                      <span className="font-semibold text-red-800 text-[18px]">Packages on Hold</span>
                                      <div className="mt-3 flex flex-row justify-center items-center">
                                        <span className="text-red-800 text-[22px] mr-3">
                                          {/* <FontAwesomeIcon icon={faPlus} /> */}
                                        </span>
                                        <span className="text-red-800 text-[36px] font-bold">18</span>
                                      </div>
                                      </div>
                                      
                                </div>

                                <div className=" p-3  text-center justify-center items-center rounded-lg flex flex-col shadow-md shadow-slate-400">
                                <iframe width="100%" height="100%" frameborder="0" marginheight="0" marginwidth="0" title="map" scrolling="no" src="https://maps.google.com/maps?width=100%&height=600&hl=en&q=%C4%B0zmir+(My%20Business%20Name)&ie=UTF8&t=&z=14&iwloc=B&output=embed" ></iframe>
                                </div>
                            </div>


                            <div className="notifify panel w-[80%] flex lg:flex-row h-auto mx-auto justify-between mt-5 flex-col">
                              
                              <div className="chart-indicators lg:w-[70%] flex flex-row h-auto justify-start rounded-lg">
                                  <div className="h-full w-full justify-center text-center rounded-lg flex flex-col">
                                        
                                        <div className=" flex flex-row justify-center items-center">
                                          <MonthlyProgress/>
                                        </div>
                                  </div>
                              </div>


                              <div className="chart-indicators lg:w-[25%]  flex flex-row h-auto  justify-end  rounded-lg mt-5 lg:mt-0">
                                  
                                  <div className="h-full w-full p-3 rounded-lg flex flex-col bg-white">
                                        
                                        <div className=" flex flex-col leading-10">
                                          <table>
                                            <tr>
                                              <td>
                                                Remain Inquiries
                                              </td>
                                              <td>
                                                4
                                              </td>
                                            </tr>

                                            <tr>
                                              <td>
                                                Returned Packages
                                              </td>
                                              <td>
                                                1
                                              </td>
                                            </tr>

                                            <tr>
                                              <td>
                                                Dispatched Packages
                                              </td>
                                              <td>
                                                13
                                              </td>
                                            </tr>

                                            <tr>
                                              <td>
                                                Onhold Packages
                                              </td>
                                              <td>
                                                9
                                              </td>
                                            </tr>




                                          </table>

                                          
                                        </div>
                                        <div className="w-full h-[100%] justify-center items-center p-3 bg-red-700 flex flex-col">
                                              <span className="text-white"> On going Deliveries </span>
                                              <span className="text-white text-[24pt] font-semibold"> 12 </span>
                                          </div>
                                  </div>
                              </div>

                              


                            </div>


                    <div className="bg-white mt-5 w-[80%] ml-[10%] overflow-x-auto rounded-lg rounded-r-lg">
                    <table className="w-[100%] text-left text-sm font-light rounded-lg rounded-r-lg">
                      <thead class="border-b font-medium dark:border-neutral-500">
                        <tr>
                          <th scope="col" class="px-6 py-4">Order ID</th>
                          <th scope="col" class="px-6 py-4">Destination</th>
                          <th scope="col" class="px-6 py-4">Vehicle</th>
                          <th scope="col" class="px-6 py-4">Status</th>
                          <th scope="col" class="px-6 py-4">Departure Date</th>
                          <th scope="col" class="px-6 py-4">Estimate Date</th>
                          <th scope="col" class="px-6 py-4">Action</th>
                        </tr>
                      </thead>
                      <tbody>

                        
                      <tr
                          class="border-b transition duration-300 ease-in-out hover:bg-neutral-100 dark:border-neutral-500 dark:hover:bg-neutral-600">
                          <td class="whitespace-nowrap px-6 py-4 font-medium">1</td>
                          <td class="whitespace-nowrap px-6 py-4">John Doe</td>
                          <td class="whitespace-nowrap px-6 py-4">Test</td>
                          <td class="whitespace-nowrap px-6 py-4">Test</td>
                          <td class="whitespace-nowrap px-6 py-4">Test</td>
                          <td class="whitespace-nowrap px-6 py-4">Test</td>
                        <td class="whitespace-nowrap px-6 py-4">
                            <Link to={"/UpdateDelivery/01"}>
                              <button class="group relative h-8 w-24 overflow-hidden rounded-2xl bg-purple-900 text-sm font-bold text-white mr-4">
                                Update
                                <div class="absolute inset-0 h-full w-full scale-0 rounded-2xl transition-all duration-300 group-hover:scale-100 group-hover:bg-white/30"></div>
                              </button>
                            </Link>
                            <Link to={""}>
                              <button class="group relative h-8 w-24 overflow-hidden rounded-2xl bg-red-500 text-sm font-bold text-white mr-4">
                                Remove
                                <div class="absolute inset-0 h-full w-full scale-0 rounded-2xl transition-all duration-300 group-hover:scale-100 group-hover:bg-white/30"></div>
                              </button>
                            </Link>
                            <Link to={"/ViewDelivery/01"}>
                              <button class="group relative h-8 w-24 overflow-hidden rounded-2xl bg-blue-500 text-sm font-bold text-white mr-4">
                                View
                                <div class="absolute inset-0 h-full w-full scale-0 rounded-2xl transition-all duration-300 group-hover:scale-100 group-hover:bg-white/30"></div>
                              </button>
                            </Link>
                          </td>
                        </tr>

                        <tr
                          class="border-b transition duration-300 ease-in-out hover:bg-neutral-100 dark:border-neutral-500 dark:hover:bg-neutral-600">
                          <td class="whitespace-nowrap px-6 py-4 font-medium">2</td>
                          <td class="whitespace-nowrap px-6 py-4"> Test</td>
                          <td class="whitespace-nowrap px-6 py-4">Test</td>
                          <td class="whitespace-nowrap px-6 py-4">Test</td>
                          <td class="whitespace-nowrap px-6 py-4">Test</td>
                          <td class="whitespace-nowrap px-6 py-4">Test</td>
                          <td class="whitespace-nowrap px-6 py-4">
                          <Link to={"/UpdateDelivery/11"}>
                              <button class="group relative h-8 w-24 overflow-hidden rounded-2xl bg-purple-900 text-sm font-bold text-white mr-4">
                                Update
                                <div class="absolute inset-0 h-full w-full scale-0 rounded-2xl transition-all duration-300 group-hover:scale-100 group-hover:bg-white/30"></div>
                              </button>
                            </Link>
                            <Link to={""}>
                              <button class="group relative h-8 w-24 overflow-hidden rounded-2xl bg-red-500 text-sm font-bold text-white mr-4">
                                Remove
                                <div class="absolute inset-0 h-full w-full scale-0 rounded-2xl transition-all duration-300 group-hover:scale-100 group-hover:bg-white/30"></div>
                              </button>
                            </Link>
                            <Link to={"/ViewDelivery/11"}>
                              <button class="group relative h-8 w-24 overflow-hidden rounded-2xl bg-blue-500 text-sm font-bold text-white mr-4">
                                View
                                <div class="absolute inset-0 h-full w-full scale-0 rounded-2xl transition-all duration-300 group-hover:scale-100 group-hover:bg-white/30"></div>
                              </button>
                            </Link>
                          </td>
                        </tr>

                        


                      </tbody>
                    </table>
                              </div>



                          </div>
                        </div>

                        
                        
                    </div>
            </div>

    );
}


export default Dashboard;


