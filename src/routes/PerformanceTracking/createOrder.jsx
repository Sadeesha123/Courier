import { useState } from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faStarHalfStroke } from '@fortawesome/free-solid-svg-icons'
import Sidepanel from "../../components/sidepanel";
import bg from '../../images/mainbg1.jpg';


function CreateOrder() {

    const [showAlert, setShowAlert] = useState(false);


    const handleSubmit = (event) => {
        event.preventDefault();
        setShowAlert(true);

        setTimeout(() => {
            setShowAlert(false);
        }, 3000);

    };



 

    return (

        <div className="main-body h-screen w-full bg-slate-100">
             <img src={bg} alt="" srcset="" className="object-cover w-[100%] h-[100%] fixed" />
             <div className="main-body-container w-full flex flex-row absolute">
                <Sidepanel />
                <div className="w-5/6 side-panel p-5 md:ml-[300px] ml-16">
                    
                    <div className="flex flex-col items-center justify-center gap-5 p-5 common-body">

                        <div className="form-body w-[60%] flex flex-col p-5 mx-auto items-center justify-center bg-white rounded-lg shadow-md shadow-slate-300">

                            <form onSubmit={handleSubmit} className="flex flex-col w-full">

               

                                <h1 className="flex items-center justify-center p-5 font-semibold uppercase text-md">Sender Details</h1>


                                <div className="flex justify-between">
                                    <label htmlFor="sendername" className="p-1 mb-2 font-semibold text-gray-600">
                                        Sender Name :
                                    </label>
                                    <input type="text" id="sendername" name="sendername" className="mb-2 border border-gray-300 rounded-lg w-[60%] " required/>
                                </div>

                                <div className="flex justify-between">
                                    <label htmlFor="sendertype" className="p-1 mb-2 font-semibold text-gray-600 ">
                                        Sender Type :
                                    </label>
                                    <select id="sendertype" name="sendertype" className="p-1 mb-2 border border-gray-300 rounded-lg w-[60%] h-[35px]">
                                        <option value="car">Normal</option>
                                        <option value="bike">Business</option>
                                    </select>
                                </div>

                                <div className="flex justify-between">
                                    <label htmlFor="sendermobile" className="p-1 mb-2 font-semibold text-gray-600">
                                        Mobile Number :
                                    </label>
                                    <input type="text" id="sendermobile" name="sendermobile" className="mb-2 border border-gray-300 rounded-lg w-[60%] h-[35px]" required />
                                </div>

                                <div className="flex justify-between">
                                    <label htmlFor="itemtype" className="p-1 mb-2 font-semibold text-gray-600">
                                        Item Type :
                                    </label>
                                    <select id="itemtype" name="itemtype" className="p-2 mb-2 border border-gray-300 rounded-lg w-[60%] h-[35px]">
                                        <option value="car">Food</option>
                                        <option value="bike">Electronic Item</option>
                                        <option value="truck">Apparel</option>
                                        <option value="car">Document</option>
                                    </select>
                                </div>

                                <div className="flex justify-between">
                                    <label htmlFor="noofpices" className="p-1 mb-2 font-semibold text-gray-600">
                                        No of Pices :
                                    </label>
                                    <input type="text" id="noofpices" name="noofpices" className="mb-2 border border-gray-300 rounded-lg w-[60%] h-[35px]" required/>
                                </div>


                                <div className="flex justify-between">
                                    <label htmlFor="date" className="p-1 mb-2 font-semibold text-gray-600">
                                        Date :
                                    </label>
                                    <input type="date" id="date" name="date" className="p-2 mb-2 border border-gray-300 rounded-lg w-[60%] h-[35px]" required />
                                </div>

                                <div className="flex justify-between">
                                    <label htmlFor="predays" className="p-1 mb-2 font-semibold text-gray-600">
                                        Predetermines Days :
                                    </label>
                                    <input type="text" id="predays" name="predays" className="p-2 mb-2 border border-gray-300 rounded-lg w-[60%] h-[35px]"  required />
                                </div>

                            </form>
                        </div>

                
                        <div className="form-body w-[60%] flex flex-col p-5 mx-auto items-center justify-center bg-white rounded-lg shadow-md shadow-slate-300">
                            <form onSubmit={handleSubmit} className="flex flex-col w-full">
                                <h1 className="flex items-center justify-center p-5 uppercase text-md font-semibold">Reciever Details</h1>



                                <div className="flex justify-between">
                                    <label htmlFor="recievername" className="p-1 mb-2 font-semibold text-gray-600">
                                        Reciever name :
                                    </label>
                                    <input type="text" id="recievername" name="recievername" className="mb-2 border border-gray-300 rounded-lg w-[60%] "  required />
                                </div>


                                <div className="flex justify-between">
                                    <label htmlFor="address" className="p-1 mb-2 font-semibold text-gray-600">
                                        Address :
                                    </label>
                                    <input type="text" id="address" name="address" className="mb-2 border border-gray-300 rounded-lg w-[60%] h-[35px]" required />
                                </div>

                                <div className="flex justify-between">
                                    <label htmlFor="province" className="mb-2 font-semibold text-gray-600">
                                        Province :
                                    </label>
                                    <select id="province" name="province" className="p-1 mb-2 border border-gray-300 rounded-lg w-[60%] h-[35px]">
                                        <option value="car">Central</option>
                                        <option value="bike">Eastern</option>
                                        <option value="truck">North Central</option>
                                        <option value="car">Northern</option>
                                        <option value="bike">North Western</option>
                                        <option value="truck">Sabaragamuwa</option>
                                        <option value="car">Southern</option>
                                        <option value="bike">Uva</option>
                                        <option value="truck">Western</option>
                                    </select>
                                </div>

                                <div className="flex justify-between">
                                    <label htmlFor="email" className="p-1 mb-2 font-semibold text-gray-600">
                                        Email :
                                    </label>
                                    <input type="text" id="email" name="email" className="mb-2 border border-gray-300 rounded-lg w-[60%] h-[35px]"  required/>
                                </div>

                                <div className="flex justify-between">
                                    <label htmlFor="recievernumber" className="p-1 mb-2 font-semibold text-gray-600">
                                        Mobile Number :
                                    </label>
                                    <input type="text" id="recievernumber" name="recievernumber" className="mb-2 border border-gray-300 rounded-lg w-[60%] h-[35px]"  required/>
                                </div>

                            </form>
                        

                        <button type="submit" className="py-2 px-4 bg-gradient-to-r from-blue-500 to-blue-700 text-white rounded-lg w-52 ml-auto mt-8" onClick={handleSubmit}>
                            Create
                        </button>

                        </div>

                    
            {/* {showAlert && (
                <div className="gap-4 fixed top-0 bottom-0 right-0 flex flex-col items-center justify-center bg-opacity-50 left-[270px] bg-black/10">
                    <div className="p-4 rounded-lg shadow-md bg-blue-500/60">
                        <p className="text-white ">Successfully Added!</p>
                    </div>
                    <img className="w-24 h-24 opacity-50" src="/images/success_alert.png" alt=""></img>
                </div>
            )} */}
                    
                    </div>
                </div>
            </div>
                    

        </div>
        

    );
}


export default CreateOrder;